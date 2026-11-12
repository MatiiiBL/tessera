// The Metal runtime of Tessera: a C ABI over Metal, loaded from Python with ctypes.
//
// Every object handed to Python is a retained Objective-C object cast to void*;
// Python releases it with tm_release. Functions that can fail return NULL or a
// negative status and copy a readable message into the caller's error buffer.
//
// A program is a list of kernel launches. tm_program_run encodes all of them into
// one command buffer and one concurrent compute encoder, with a memory barrier
// before each launch that the caller marked as depending on an earlier one.
// tm_program_profile encodes one compute pass per launch instead, so that the GPU
// can timestamp the start and the end of each kernel.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <dispatch/dispatch.h>
#include <mach/mach_time.h>
#include <stdint.h>
#include <string.h>

#define TM_API __attribute__((visibility("default")))

enum {
    TM_OK = 0,
    TM_ERROR = -1,
    TM_TIMEOUT = -2,
};

static void tm_copy_error(char *err, size_t cap, NSString *text) {
    if (err == NULL || cap == 0) return;
    const char *utf8 = text.UTF8String ? text.UTF8String : "unknown error";
    strncpy(err, utf8, cap - 1);
    err[cap - 1] = '\0';
}

static double tm_now_seconds(void) {
    static mach_timebase_info_data_t timebase;
    if (timebase.denom == 0) mach_timebase_info(&timebase);
    return (double)mach_absolute_time() * timebase.numer / timebase.denom * 1e-9;
}

// -- device ------------------------------------------------------------------------

@interface TMContext : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) id<MTLCommandQueue> queue;
@property(nonatomic, strong) id<MTLCounterSet> timestamps;  // nil without stage counters
@end

@implementation TMContext
@end

typedef struct {
    char name[128];
    uint64_t max_threads_per_threadgroup;
    uint64_t max_threadgroup_memory;
    uint64_t max_buffer_length;
    uint64_t recommended_working_set;
    int32_t has_unified_memory;
    int32_t has_stage_timestamps;
} tm_device_info;

TM_API void tm_release(void *object) {
    if (object != NULL) CFRelease(object);
}

TM_API void *tm_context_create(void) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) return NULL;
        TMContext *context = [TMContext new];
        context.device = device;
        context.queue = [device newCommandQueue];
        if (context.queue == nil) return NULL;
        if ([device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]) {
            for (id<MTLCounterSet> set in device.counterSets) {
                if ([set.name isEqualToString:MTLCommonCounterSetTimestamp]) {
                    context.timestamps = set;
                }
            }
        }
        return (void *)CFBridgingRetain(context);
    }
}

TM_API void tm_context_info(void *handle, tm_device_info *out) {
    @autoreleasepool {
        TMContext *context = (__bridge TMContext *)handle;
        id<MTLDevice> device = context.device;
        memset(out, 0, sizeof(*out));
        strncpy(out->name, device.name.UTF8String, sizeof(out->name) - 1);
        out->max_threads_per_threadgroup = device.maxThreadsPerThreadgroup.width;
        out->max_threadgroup_memory = device.maxThreadgroupMemoryLength;
        out->max_buffer_length = device.maxBufferLength;
        out->recommended_working_set = device.recommendedMaxWorkingSetSize;
        out->has_unified_memory = device.hasUnifiedMemory ? 1 : 0;
        out->has_stage_timestamps = context.timestamps != nil ? 1 : 0;
    }
}

// -- libraries and pipelines -------------------------------------------------------

TM_API void *tm_library_create(void *handle, const char *source, char *err, size_t cap) {
    @autoreleasepool {
        TMContext *context = (__bridge TMContext *)handle;
        MTLCompileOptions *options = [MTLCompileOptions new];
        options.languageVersion = MTLLanguageVersion3_1;
        if (@available(macOS 15.0, *)) {
            // Fast arithmetic, but infinities and NaNs keep their meaning: reductions
            // start from -INFINITY.
            options.mathMode = MTLMathModeRelaxed;
        }
        NSError *error = nil;
        NSString *text = [NSString stringWithUTF8String:source];
        id<MTLLibrary> library = [context.device newLibraryWithSource:text
                                                              options:options
                                                                error:&error];
        if (library == nil) {
            tm_copy_error(err, cap, error ? error.localizedDescription : @"compilation failed");
            return NULL;
        }
        return (void *)CFBridgingRetain(library);
    }
}

typedef struct {
    uint64_t max_total_threads;
    uint64_t thread_execution_width;
    uint64_t static_threadgroup_memory;
} tm_pipeline_info;

TM_API void *tm_pipeline_create(void *handle, void *library_handle, const char *name, char *err,
                                size_t cap) {
    @autoreleasepool {
        TMContext *context = (__bridge TMContext *)handle;
        id<MTLLibrary> library = (__bridge id<MTLLibrary>)library_handle;
        id<MTLFunction> function = [library newFunctionWithName:@(name)];
        if (function == nil) {
            tm_copy_error(err, cap,
                          [NSString stringWithFormat:@"no kernel named %s in the library", name]);
            return NULL;
        }
        NSError *error = nil;
        id<MTLComputePipelineState> pipeline =
            [context.device newComputePipelineStateWithFunction:function error:&error];
        if (pipeline == nil) {
            tm_copy_error(err, cap, error ? error.localizedDescription : @"pipeline failed");
            return NULL;
        }
        return (void *)CFBridgingRetain(pipeline);
    }
}

TM_API void tm_pipeline_info_get(void *handle, tm_pipeline_info *out) {
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)handle;
    out->max_total_threads = pipeline.maxTotalThreadsPerThreadgroup;
    out->thread_execution_width = pipeline.threadExecutionWidth;
    out->static_threadgroup_memory = pipeline.staticThreadgroupMemoryLength;
}

// -- buffers -----------------------------------------------------------------------

TM_API void *tm_buffer_create(void *handle, uint64_t nbytes) {
    @autoreleasepool {
        TMContext *context = (__bridge TMContext *)handle;
        // Shared storage is the unified memory the CPU sees, so the host reads and
        // writes the contents directly. Programs order their own launches, so Metal's
        // hazard tracking would only add cost.
        MTLResourceOptions options =
            MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked;
        id<MTLBuffer> buffer = [context.device newBufferWithLength:(nbytes ? nbytes : 16)
                                                           options:options];
        if (buffer == nil) return NULL;
        return (void *)CFBridgingRetain(buffer);
    }
}

TM_API void *tm_buffer_contents(void *handle) {
    return [(__bridge id<MTLBuffer>)handle contents];
}

TM_API uint64_t tm_buffer_length(void *handle) {
    return [(__bridge id<MTLBuffer>)handle length];
}

// -- programs ----------------------------------------------------------------------

@interface TMLaunch : NSObject {
   @public
    MTLSize grid;
    MTLSize threads;
    NSUInteger count;
    __strong id<MTLBuffer> *buffers;
    NSUInteger *offsets;
    BOOL barrier;
}
@property(nonatomic, strong) id<MTLComputePipelineState> pipeline;
@property(nonatomic, copy) NSString *label;
@end

@implementation TMLaunch
- (void)dealloc {
    for (NSUInteger i = 0; i < count; i++) buffers[i] = nil;
    free(buffers);
    free(offsets);
}
@end

@interface TMProgram : NSObject
@property(nonatomic, strong) TMContext *context;
@property(nonatomic, strong) NSMutableArray<TMLaunch *> *launches;
@end

@implementation TMProgram
@end

TM_API void *tm_program_create(void *handle) {
    @autoreleasepool {
        TMProgram *program = [TMProgram new];
        program.context = (__bridge TMContext *)handle;
        program.launches = [NSMutableArray new];
        return (void *)CFBridgingRetain(program);
    }
}

TM_API int tm_program_add(void *handle, void *pipeline, const char *label, const uint32_t grid[3],
                          const uint32_t threads[3], uint32_t count, void *const *buffers,
                          const uint64_t *offsets, int32_t barrier) {
    @autoreleasepool {
        TMProgram *program = (__bridge TMProgram *)handle;
        TMLaunch *launch = [TMLaunch new];
        launch.pipeline = (__bridge id<MTLComputePipelineState>)pipeline;
        launch.label = @(label);
        launch->grid = MTLSizeMake(grid[0], grid[1], grid[2]);
        launch->threads = MTLSizeMake(threads[0], threads[1], threads[2]);
        launch->count = count;
        launch->buffers = (__strong id<MTLBuffer> *)calloc(count ? count : 1, sizeof(id));
        launch->offsets = (NSUInteger *)calloc(count ? count : 1, sizeof(NSUInteger));
        if (launch->buffers == NULL || launch->offsets == NULL) return TM_ERROR;
        for (uint32_t i = 0; i < count; i++) {
            launch->buffers[i] = (__bridge id<MTLBuffer>)buffers[i];
            launch->offsets[i] = (NSUInteger)offsets[i];
        }
        launch->barrier = barrier != 0;
        [program.launches addObject:launch];
        return (int)program.launches.count - 1;
    }
}

TM_API uint32_t tm_program_size(void *handle) {
    return (uint32_t)((__bridge TMProgram *)handle).launches.count;
}

static void tm_encode(id<MTLComputeCommandEncoder> encoder, TMLaunch *launch) {
    [encoder setComputePipelineState:launch.pipeline];
    if (launch->count > 0) {
        [encoder setBuffers:launch->buffers
                    offsets:launch->offsets
                  withRange:NSMakeRange(0, launch->count)];
    }
    [encoder dispatchThreadgroups:launch->grid threadsPerThreadgroup:launch->threads];
}

static NSString *tm_describe_failure(id<MTLCommandBuffer> buffer) {
    NSError *error = buffer.error;
    if (error == nil) return @"the command buffer failed";
    NSMutableString *text = [NSMutableString stringWithString:error.localizedDescription];
    NSArray *infos = error.userInfo[MTLCommandBufferEncoderInfoErrorKey];
    for (id<MTLCommandBufferEncoderInfo> info in infos) {
        if (info.errorState == MTLCommandEncoderErrorStateFaulted) {
            [text appendFormat:@" (in %@)", info.label];
        }
    }
    return text;
}

// Commits and waits at most `timeout` seconds (0: forever). The completion handler
// keeps `program` alive in case the caller gives up waiting.
static int tm_commit_and_wait(id<MTLCommandBuffer> buffer, TMProgram *program, double timeout,
                              char *err, size_t cap) {
    dispatch_semaphore_t done = dispatch_semaphore_create(0);
    __block TMProgram *keep = program;
    [buffer addCompletedHandler:^(id<MTLCommandBuffer> finished) {
      (void)finished;
      keep = nil;
      dispatch_semaphore_signal(done);
    }];
    [buffer commit];
    dispatch_time_t limit = timeout > 0
                                ? dispatch_time(DISPATCH_TIME_NOW, (int64_t)(timeout * 1e9))
                                : DISPATCH_TIME_FOREVER;
    if (dispatch_semaphore_wait(done, limit) != 0) {
        tm_copy_error(err, cap,
                      [NSString stringWithFormat:@"the GPU did not finish within %.3g s", timeout]);
        return TM_TIMEOUT;
    }
    if (buffer.status == MTLCommandBufferStatusError) {
        tm_copy_error(err, cap, tm_describe_failure(buffer));
        return TM_ERROR;
    }
    return TM_OK;
}

static id<MTLCommandBuffer> tm_command_buffer(TMProgram *program, BOOL detailed_errors) {
    MTLCommandBufferDescriptor *descriptor = [MTLCommandBufferDescriptor new];
    // The program holds every pipeline and buffer it uses.
    descriptor.retainedReferences = NO;
    if (detailed_errors) {
        descriptor.errorOptions = MTLCommandBufferErrorOptionEncoderExecutionStatus;
    }
    return [program.context.queue commandBufferWithDescriptor:descriptor];
}

// Runs the whole program `repeat` times, one command buffer each, and reports each
// run's wall-clock time (encoding to completion) and GPU time. `serial` uses a
// serial encoder (Metal orders every launch) instead of barriers where marked.
TM_API int tm_program_run(void *handle, uint32_t repeat, int32_t serial, double timeout,
                          double *wall_seconds, double *gpu_seconds, char *err, size_t cap) {
    TMProgram *program = (__bridge TMProgram *)handle;
    for (uint32_t r = 0; r < repeat; r++) {
        @autoreleasepool {
            double start = tm_now_seconds();
            id<MTLCommandBuffer> buffer = tm_command_buffer(program, NO);
            MTLDispatchType type = serial ? MTLDispatchTypeSerial : MTLDispatchTypeConcurrent;
            id<MTLComputeCommandEncoder> encoder =
                [buffer computeCommandEncoderWithDispatchType:type];
            BOOL first = YES;
            for (TMLaunch *launch in program.launches) {
                if (launch->barrier && !first && !serial) {
                    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                }
                tm_encode(encoder, launch);
                first = NO;
            }
            [encoder endEncoding];
            int status = tm_commit_and_wait(buffer, program, timeout, err, cap);
            if (status != TM_OK) return status;
            if (wall_seconds) wall_seconds[r] = tm_now_seconds() - start;
            if (gpu_seconds) gpu_seconds[r] = buffer.GPUEndTime - buffer.GPUStartTime;
        }
    }
    return TM_OK;
}

// Times each launch on the GPU. With stage-boundary timestamps (Apple GPUs), every
// launch is its own compute pass inside one command buffer, ordered by a fence, and
// the GPU samples a timestamp at the start and at the end of each pass. Without
// them, each launch gets its own command buffer, timed by GPUStartTime/GPUEndTime.
// `seconds` receives one value per launch.
TM_API int tm_program_profile(void *handle, double timeout, double *seconds, char *err,
                              size_t cap) {
    TMProgram *program = (__bridge TMProgram *)handle;
    NSUInteger n = program.launches.count;
    if (n == 0) return TM_OK;
    @autoreleasepool {
        TMContext *context = program.context;
        if (context.timestamps == nil) {
            for (NSUInteger i = 0; i < n; i++) {
                id<MTLCommandBuffer> buffer = tm_command_buffer(program, YES);
                id<MTLComputeCommandEncoder> encoder = [buffer computeCommandEncoder];
                encoder.label = program.launches[i].label;
                tm_encode(encoder, program.launches[i]);
                [encoder endEncoding];
                int status = tm_commit_and_wait(buffer, program, timeout, err, cap);
                if (status != TM_OK) return status;
                seconds[i] = buffer.GPUEndTime - buffer.GPUStartTime;
            }
            return TM_OK;
        }

        MTLCounterSampleBufferDescriptor *descriptor = [MTLCounterSampleBufferDescriptor new];
        descriptor.counterSet = context.timestamps;
        descriptor.storageMode = MTLStorageModeShared;
        descriptor.sampleCount = 2 * n;
        NSError *error = nil;
        id<MTLCounterSampleBuffer> samples =
            [context.device newCounterSampleBufferWithDescriptor:descriptor error:&error];
        if (samples == nil) {
            tm_copy_error(err, cap, error ? error.localizedDescription : @"no sample buffer");
            return TM_ERROR;
        }
        id<MTLFence> fence = [context.device newFence];
        id<MTLCommandBuffer> buffer = tm_command_buffer(program, YES);
        MTLTimestamp cpu0 = 0, gpu0 = 0, cpu1 = 0, gpu1 = 0;
        [context.device sampleTimestamps:&cpu0 gpuTimestamp:&gpu0];
        for (NSUInteger i = 0; i < n; i++) {
            MTLComputePassDescriptor *pass = [MTLComputePassDescriptor computePassDescriptor];
            pass.dispatchType = MTLDispatchTypeSerial;
            pass.sampleBufferAttachments[0].sampleBuffer = samples;
            pass.sampleBufferAttachments[0].startOfEncoderSampleIndex = 2 * i;
            pass.sampleBufferAttachments[0].endOfEncoderSampleIndex = 2 * i + 1;
            id<MTLComputeCommandEncoder> encoder = [buffer computeCommandEncoderWithDescriptor:pass];
            encoder.label = program.launches[i].label;
            if (i > 0) [encoder waitForFence:fence];
            tm_encode(encoder, program.launches[i]);
            [encoder updateFence:fence];
            [encoder endEncoding];
        }
        int status = tm_commit_and_wait(buffer, program, timeout, err, cap);
        if (status != TM_OK) return status;
        [context.device sampleTimestamps:&cpu1 gpuTimestamp:&gpu1];

        NSData *data = [samples resolveCounterRange:NSMakeRange(0, 2 * n)];
        if (data == nil || data.length < 2 * n * sizeof(MTLCounterResultTimestamp)) {
            tm_copy_error(err, cap, @"the GPU returned no timestamps");
            return TM_ERROR;
        }
        const MTLCounterResultTimestamp *stamps = data.bytes;
        // GPU ticks to seconds, from two (CPU, GPU) pairs sampled around the run;
        // the CPU timestamps are in nanoseconds.
        double ns_per_tick =
            gpu1 > gpu0 ? (double)(cpu1 - cpu0) / (double)(gpu1 - gpu0) : 1.0;
        for (NSUInteger i = 0; i < n; i++) {
            uint64_t begin = stamps[2 * i].timestamp, end = stamps[2 * i + 1].timestamp;
            if (begin == MTLCounterErrorValue || end == MTLCounterErrorValue || end < begin) {
                seconds[i] = -1.0;
            } else {
                seconds[i] = (double)(end - begin) * ns_per_tick * 1e-9;
            }
        }
        return TM_OK;
    }
}
