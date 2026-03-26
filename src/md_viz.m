/*
 * Real-time MD Visualization — Metal render pipeline.
 *
 * Reads particle positions directly from the simulation's shared memory.
 * Particles rendered as instanced point sprites, color-mapped by speed.
 * Blue = cold, white = thermal, red = hot.
 *
 * Usage: ./moleqular-viz [--ncells=N] [--kernel=metal|neon|omp|cl]
 */

#import <Cocoa/Cocoa.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>
#include "md_types.h"
#include "md_system.h"
#include "md_force.h"
#include "md_integrate.h"
#include <math.h>
#include <string.h>

/* Embedded Metal shaders for rendering */
static NSString *const kVizShaderSource =
    @"#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "\n"
    "struct VertexOut {\n"
    "    float4 position [[position]];\n"
    "    float  point_size [[point_size]];\n"
    "    float  speed;\n"
    "};\n"
    "\n"
    "struct VizParams {\n"
    "    float4x4 mvp;\n"
    "    float    point_size;\n"
    "    float    max_speed;\n"
    "};\n"
    "\n"
    "vertex VertexOut viz_vertex(\n"
    "    uint vid [[vertex_id]],\n"
    "    const device float *x [[buffer(0)]],\n"
    "    const device float *y [[buffer(1)]],\n"
    "    const device float *z [[buffer(2)]],\n"
    "    const device float *vx [[buffer(3)]],\n"
    "    const device float *vy [[buffer(4)]],\n"
    "    const device float *vz [[buffer(5)]],\n"
    "    constant VizParams &p [[buffer(6)]]\n"
    ") {\n"
    "    VertexOut out;\n"
    "    float4 pos = float4(x[vid], y[vid], z[vid], 1.0);\n"
    "    out.position = p.mvp * pos;\n"
    "    out.point_size = p.point_size;\n"
    "    float s = sqrt(vx[vid]*vx[vid] + vy[vid]*vy[vid] + vz[vid]*vz[vid]);\n"
    "    out.speed = clamp(s / p.max_speed, 0.0, 1.0);\n"
    "    return out;\n"
    "}\n"
    "\n"
    "fragment float4 viz_fragment(VertexOut in [[stage_in]],\n"
    "                             float2 pc [[point_coord]]) {\n"
    "    float d = length(pc - float2(0.5));\n"
    "    if (d > 0.5) discard_fragment();\n"
    "    float alpha = 1.0 - smoothstep(0.3, 0.5, d);\n"
    "    float t = in.speed;\n"
    "    float3 cold = float3(0.2, 0.4, 1.0);\n"
    "    float3 mid  = float3(0.9, 0.9, 0.9);\n"
    "    float3 hot  = float3(1.0, 0.2, 0.1);\n"
    "    float3 col = t < 0.5 ? mix(cold, mid, t*2.0) : mix(mid, hot, (t-0.5)*2.0);\n"
    "    return float4(col * alpha, alpha);\n"
    "}\n";

typedef struct {
    float mvp[16];     /* 4x4 column-major */
    float point_size;
    float max_speed;
    float _pad[2];
} VizParams;

/* Forward declarations */
static void mat4_perspective(float *out, float fovy, float aspect, float near, float far);
static void mat4_look_at(float *out, float ex, float ey, float ez,
                          float cx, float cy, float cz);
static void mat4_multiply(float *out, const float *a, const float *b);

/* --- App Delegate + Renderer --- */

@interface MDVizRenderer : NSObject <MTKViewDelegate>
@property (nonatomic) MDSystem *sys;
@property (nonatomic) ForceFunc force_func;
@property (nonatomic) int step;
@property (nonatomic) int steps_per_frame;
@property (nonatomic) float angle;
@end

@implementation MDVizRenderer {
    id<MTLDevice>               _device;
    id<MTLCommandQueue>         _queue;
    id<MTLRenderPipelineState>  _pipeline;
    id<MTLBuffer>               _buf_x, _buf_y, _buf_z;
    id<MTLBuffer>               _buf_vx, _buf_vy, _buf_vz;
    id<MTLBuffer>               _buf_params;
    int                         _buf_n;
}

- (instancetype)initWithDevice:(id<MTLDevice>)device sys:(MDSystem *)sys force:(ForceFunc)ff {
    self = [super init];
    _device = device;
    _queue = [device newCommandQueue];
    _sys = sys;
    _force_func = ff;
    _step = 0;
    _steps_per_frame = 10;
    _angle = 0.0f;

    /* Build render pipeline */
    NSError *error = nil;
    MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
    id<MTLLibrary> lib = [device newLibraryWithSource:kVizShaderSource options:opts error:&error];
    if (!lib) {
        fprintf(stderr, "[Viz] Shader compilation failed: %s\n",
                [[error localizedDescription] UTF8String]);
        return nil;
    }

    MTLRenderPipelineDescriptor *desc = [[MTLRenderPipelineDescriptor alloc] init];
    desc.vertexFunction   = [lib newFunctionWithName:@"viz_vertex"];
    desc.fragmentFunction = [lib newFunctionWithName:@"viz_fragment"];
    desc.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
    desc.colorAttachments[0].blendingEnabled = YES;
    desc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
    desc.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    desc.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
    desc.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;

    _pipeline = [device newRenderPipelineStateWithDescriptor:desc error:&error];

    _buf_params = [device newBufferWithLength:sizeof(VizParams)
                                      options:MTLResourceStorageModeShared];
    _buf_n = 0;
    return self;
}

- (void)allocBuffersForN:(int)n {
    if (n <= _buf_n) return;
    NSUInteger sz = (NSUInteger)n * sizeof(float);
    MTLResourceOptions opts = MTLResourceStorageModeShared;
    _buf_x  = [_device newBufferWithLength:sz options:opts];
    _buf_y  = [_device newBufferWithLength:sz options:opts];
    _buf_z  = [_device newBufferWithLength:sz options:opts];
    _buf_vx = [_device newBufferWithLength:sz options:opts];
    _buf_vy = [_device newBufferWithLength:sz options:opts];
    _buf_vz = [_device newBufferWithLength:sz options:opts];
    _buf_n = n;
}

- (void)mtkView:(MTKView *)view drawableSizeWillChange:(CGSize)size {
    /* nothing needed */
}

- (void)drawInMTKView:(MTKView *)view {
    MDSystem *sys = _sys;
    int n = sys->n_real;

    /* Advance simulation */
    for (int s = 0; s < _steps_per_frame; s++) {
        md_integrate_positions(sys, MD_DT);
        float pe;
        _force_func(sys, &pe);
        md_integrate_velocities(sys, MD_DT);
        _step++;
    }

    _angle += 0.005f;

    /* Upload positions + velocities */
    [self allocBuffersForN:n];
    memcpy([_buf_x contents],  sys->x,  (size_t)n * sizeof(float));
    memcpy([_buf_y contents],  sys->y,  (size_t)n * sizeof(float));
    memcpy([_buf_z contents],  sys->z,  (size_t)n * sizeof(float));
    memcpy([_buf_vx contents], sys->vx, (size_t)n * sizeof(float));
    memcpy([_buf_vy contents], sys->vy, (size_t)n * sizeof(float));
    memcpy([_buf_vz contents], sys->vz, (size_t)n * sizeof(float));

    /* Camera: orbit around box center */
    float half = sys->lbox * 0.5f;
    float dist = sys->lbox * 2.0f;
    float ex = half + dist * sinf(_angle);
    float ey = half + dist * 0.4f;
    float ez = half + dist * cosf(_angle);

    float proj[16], look[16], mvp[16];
    CGSize sz = view.drawableSize;
    float aspect = (float)sz.width / (float)sz.height;
    mat4_perspective(proj, 0.8f, aspect, 0.1f, dist * 4.0f);
    mat4_look_at(look, ex, ey, ez, half, half, half);
    mat4_multiply(mvp, proj, look);

    VizParams *params = (VizParams *)[_buf_params contents];
    memcpy(params->mvp, mvp, 16 * sizeof(float));
    params->point_size = (n < 2000) ? 8.0f : (n < 10000) ? 4.0f : 2.0f;
    params->max_speed = 3.0f;

    /* Render */
    MTLRenderPassDescriptor *rpd = view.currentRenderPassDescriptor;
    if (!rpd) return;

    id<MTLCommandBuffer> cmd = [_queue commandBuffer];
    id<MTLRenderCommandEncoder> enc = [cmd renderCommandEncoderWithDescriptor:rpd];

    [enc setRenderPipelineState:_pipeline];
    [enc setVertexBuffer:_buf_x      offset:0 atIndex:0];
    [enc setVertexBuffer:_buf_y      offset:0 atIndex:1];
    [enc setVertexBuffer:_buf_z      offset:0 atIndex:2];
    [enc setVertexBuffer:_buf_vx     offset:0 atIndex:3];
    [enc setVertexBuffer:_buf_vy     offset:0 atIndex:4];
    [enc setVertexBuffer:_buf_vz     offset:0 atIndex:5];
    [enc setVertexBuffer:_buf_params offset:0 atIndex:6];
    [enc drawPrimitives:MTLPrimitiveTypePoint vertexStart:0 vertexCount:(NSUInteger)n];
    [enc endEncoding];

    [cmd presentDrawable:view.currentDrawable];
    [cmd commit];

    static int _fc = 0;
    static double _ft0 = 0;
    if (_fc == 0) { struct timespec _ts; clock_gettime(CLOCK_MONOTONIC, &_ts); _ft0 = _ts.tv_sec + _ts.tv_nsec*1e-9; }
    if (++_fc % 60 == 0) {
        struct timespec _ts; clock_gettime(CLOCK_MONOTONIC, &_ts);
        double now = _ts.tv_sec + _ts.tv_nsec*1e-9;
        fprintf(stderr, "[VIZ] %.1f fps (%.1f ms/frame)\n", 60.0/(now-_ft0), (now-_ft0)/60.0*1000.0);
        _ft0 = now;
    }
}
@end

@interface MDVizAppDelegate : NSObject <NSApplicationDelegate>
@property (nonatomic) MDSystem *sys;
@property (nonatomic) ForceFunc force_func;
@end

@implementation MDVizAppDelegate {
    NSWindow       *_window;
    MDVizRenderer  *_renderer;  /* strong ref — MTKView delegate is weak */
}

- (void)applicationDidFinishLaunching:(NSNotification *)notification {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();

    /* Initial force computation */
    float pe;
    _force_func(_sys, &pe);

    /* Create window */
    NSRect frame = NSMakeRect(100, 100, 1024, 768);
    _window = [[NSWindow alloc]
        initWithContentRect:frame
                  styleMask:(NSWindowStyleMaskTitled | NSWindowStyleMaskClosable |
                            NSWindowStyleMaskResizable)
                    backing:NSBackingStoreBuffered
                      defer:NO];
    [_window setTitle:[NSString stringWithFormat:@"moleqular — %d particles", _sys->n_real]];

    MTKView *mtkView = [[MTKView alloc] initWithFrame:frame device:device];
    mtkView.colorPixelFormat = MTLPixelFormatBGRA8Unorm;
    mtkView.clearColor = MTLClearColorMake(0.05, 0.05, 0.08, 1.0);
    mtkView.preferredFramesPerSecond = 30;

    _renderer = [[MDVizRenderer alloc] initWithDevice:device
                                                 sys:_sys
                                               force:_force_func];
    mtkView.delegate = _renderer;
    _window.contentView = mtkView;

    [_window makeKeyAndOrderFront:nil];
    [NSApp activateIgnoringOtherApps:YES];
}

- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication *)sender {
    return YES;
}
@end

/* --- Entry point for visualization binary --- */
int main(int argc, char **argv) {
    ForceFunc compute_forces = md_force_neon;
    const char *mode = "NEON";
    int ncells = MD_FCC_N;

    for (int i = 1; i < argc; i++) {
        if (strncmp(argv[i], "--ncells=", 9) == 0) {
            ncells = atoi(argv[i] + 9);
        } else if (strcmp(argv[i], "--metal") == 0) {
            compute_forces = md_force_metal;
            mode = "Metal GPU";
        } else if (strcmp(argv[i], "--omp") == 0) {
            compute_forces = md_force_omp;
            mode = "OpenMP+NEON";
        } else if (strcmp(argv[i], "--cl") == 0) {
            compute_forces = md_force_neon_cl;
            mode = "NEON+CellList";
        } else if (strcmp(argv[i], "--omp-cl") == 0) {
            compute_forces = md_force_omp_cl;
            mode = "OMP+CellList";
        } else if (strcmp(argv[i], "--metal-cl") == 0) {
            compute_forces = md_force_metal_cl;
            mode = "Metal+CellList";
        }
    }

    MDSystem *sys = md_system_create(ncells, MD_DENSITY, MD_TEMP0);
    printf("moleqular-viz — %s, %d particles\n", mode, sys->n_real);

    @autoreleasepool {
        [NSApplication sharedApplication];
        [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];

        MDVizAppDelegate *delegate = [[MDVizAppDelegate alloc] init];
        delegate.sys = sys;
        delegate.force_func = compute_forces;
        [NSApp setDelegate:delegate];
        [NSApp run];
    }

    md_system_destroy(sys);
    return 0;
}

/* --- Matrix math --- */

static void mat4_perspective(float *m, float fovy, float aspect, float near, float far) {
    memset(m, 0, 16 * sizeof(float));
    float f = 1.0f / tanf(fovy * 0.5f);
    m[0]  = f / aspect;
    m[5]  = f;
    m[10] = (far + near) / (near - far);
    m[11] = -1.0f;
    m[14] = (2.0f * far * near) / (near - far);
}

static void mat4_look_at(float *m, float ex, float ey, float ez,
                          float cx, float cy, float cz) {
    /* Forward = normalize(center - eye) */
    float fx = cx - ex, fy = cy - ey, fz = cz - ez;
    float len = sqrtf(fx*fx + fy*fy + fz*fz);
    fx /= len; fy /= len; fz /= len;

    /* Right = normalize(forward × up), up = (0,1,0) */
    /* cross(f, up) = (fy*0 - fz*1, fz*0 - fx*0, fx*1 - fy*0) */
    float sx = -fz, sy = 0.0f, sz = fx;
    len = sqrtf(sx*sx + sy*sy + sz*sz);
    if (len > 1e-6f) { sx /= len; sy /= len; sz /= len; }

    /* Up = normalize(right × forward) */
    float ux = sy*fz - sz*fy;
    float uy = sz*fx - sx*fz;
    float uz = sx*fy - sy*fx;

    /* Column-major view matrix */
    m[0]  =  sx; m[1]  =  ux; m[2]  = -fx; m[3]  = 0.0f;
    m[4]  =  sy; m[5]  =  uy; m[6]  = -fy; m[7]  = 0.0f;
    m[8]  =  sz; m[9]  =  uz; m[10] = -fz; m[11] = 0.0f;
    m[12] = -(sx*ex + sy*ey + sz*ez);
    m[13] = -(ux*ex + uy*ey + uz*ez);
    m[14] =  (fx*ex + fy*ey + fz*ez);
    m[15] = 1.0f;
}

static void mat4_multiply(float *out, const float *a, const float *b) {
    float tmp[16];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
            tmp[j*4+i] = 0;
            for (int k = 0; k < 4; k++)
                tmp[j*4+i] += a[k*4+i] * b[j*4+k];
        }
    memcpy(out, tmp, 16 * sizeof(float));
}
