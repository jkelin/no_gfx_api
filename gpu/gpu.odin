
package gpu

import "core:slice"
import "base:runtime"
import intr "base:intrinsics"

import sdl "vendor:sdl3"
import vk "vendor:vulkan"

// This API follows the ZII (Zero Is Initialization) principle. Initializing to 0
// will yield predictable and reasonable behavior in general.

// Handles
Handle :: rawptr
Texture_Handle :: distinct Handle
Command_Buffer :: distinct Handle
Semaphore :: distinct Handle
Shader :: distinct Handle
BVH :: struct { _: Handle }
Texture_Descriptor :: struct { bytes: [8]u64 }
Sampler_Descriptor :: struct { bytes: [4]u64 }
BVH_Descriptor :: struct { bytes: [4]u64 }

// Enums
Feature :: enum { Raytracing = 0 }
Features :: bit_set[Feature; u32]
Allocation_Type :: enum { Default = 0, Descriptors, Texture }
Memory :: enum { Default = 0, GPU, Readback }
Queue :: enum { Main = 0, Compute, Transfer }
Texture_Type :: enum { D2 = 0, D3, D1 }
Texture_Format :: enum {
	Default = 0,
	RGBA8_Unorm,
	BGRA8_Unorm,
	RGBA8_SRGB,
	D32_Float,
	RGBA16_Float,
	RGBA32_Float,
	BC1_RGBA_Unorm,
	BC3_RGBA_Unorm,
	BC7_RGBA_Unorm,
	ASTC_4x4_RGBA_Unorm,
	ETC2_RGB8_Unorm,
	ETC2_RGBA8_Unorm,
	EAC_R11_Unorm,
	EAC_RG11_Unorm,
}
Usage :: enum { Sampled = 0, Storage, Transfer_Src, Color_Attachment, Depth_Stencil_Attachment }
Usage_Flags :: bit_set[Usage; u32]
Shader_Type_Graphics :: enum { Vertex = 0, Fragment }
Load_Op :: enum { Clear = 0, Load, Dont_Care }
Store_Op :: enum { Store = 0, Dont_Care }
Compare_Op :: enum { Never = 0, Less, Equal, Less_Equal, Greater, Not_Equal, Greater_Equal, Always }
Blend_Op :: enum { Add, Subtract, Rev_Subtract, Min, Max }
Blend_Factor :: enum { Zero, One, Src_Color, Dst_Color, Src_Alpha }
Depth_Mode :: enum { Read = 0, Write }
Depth_Flags :: bit_set[Depth_Mode; u32]
Hazard :: enum { Draw_Arguments = 0, Descriptors, Depth_Stencil, BVHs }
Hazard_Flags :: bit_set[Hazard; u32]
Stage :: enum { Transfer = 0, Compute, Raster_Color_Out, Fragment_Shader, Vertex_Shader, Build_BVH, All }
Color_Component_Flag :: enum { R = 0, G = 1, B = 2, A = 3 }
Color_Component_Flags :: distinct bit_set[Color_Component_Flag; u8]
Filter :: enum { Linear = 0, Nearest }
Address_Mode :: enum { Repeat = 0, Mirrored_Repeat, Clamp_To_Edge }
BVH_Instance_Flag :: enum { Disable_Culling = 0, Flip_Facing = 1, Force_Opaque = 2, Force_Not_Opaque = 3 }
BVH_Instance_Flags :: distinct bit_set[BVH_Instance_Flag; u32]
BVH_Opacity :: enum { Fully_Opaque = 0, Transparent }
BVH_Hint :: enum { Default = 0, Prefer_Fast_Trace, Prefer_Fast_Build, Prefer_Low_Memory }
BVH_Capability :: enum { Update = 0, Compaction }
BVH_Capabilities :: distinct bit_set[BVH_Capability; u32]

// Constants
All_Mips: u8 : max(u8)
All_Layers: u16 : max(u16)

// Structs

Blit_Rect :: struct
{
    offset_a: [3]i32,  // offset_a == 0 && offset_b == 0 -> full image
    offset_b: [3]i32,  // offset_a == 0 && offset_b == 0 -> full image
    mip_level: u32,
    base_layer: u32,
    layer_count: u32,
}

Mip_Copy_Region :: struct {
	src_offset:  u64, // Offset in staging buffer
	mip_level:   u32,
	array_layer: u32,
	layer_count: u32,
}

Texture_Desc :: struct
{
    type: Texture_Type,
    dimensions: [3]u32,
    mip_count: u32,     // 0 = 1
    layer_count: u32,   // 0 = 1
    sample_count: u32,  // 0 = 1
    format: Texture_Format,
    usage: Usage_Flags,
}

Sampler_Desc :: struct
{
    min_filter: Filter,
    mag_filter: Filter,
    mip_filter: Filter,
    address_mode_u: Address_Mode,
    address_mode_v: Address_Mode,
    address_mode_w: Address_Mode,
    mip_lod_bias: f32,
    min_lod: f32,
    max_lod: f32,  // 0.0 = use all lods
    max_anisotropy: f32,
}

Texture_View_Desc :: struct
{
    type: Texture_Type,
    format: Texture_Format,  // .Default = inherits the texture's format
    base_mip: u32,
    mip_count: u8,     // 0 = All_Mips
    base_layer: u16,
    layer_count: u16,  // 0 = All_Layers
}

Render_Attachment :: struct
{
    texture: Texture,
    view: Texture_View_Desc,
    load_op: Load_Op,
    store_op: Store_Op,
    clear_color: [4]f32,
}

Render_Pass_Desc :: struct
{
    render_area_offset: [2]i32,
    render_area_size:   [2]u32,  // 0 = full texture size
    layer_count:        u32,     // 0 = 1
    view_mask:          u32,
    color_attachments:  []Render_Attachment,
    depth_attachment:   Maybe(Render_Attachment),
    stencil_attachment: Maybe(Render_Attachment),
}

Texture :: struct #all_or_none
{
    dimensions: [3]u32,
    format: Texture_Format,
    mip_count: u32,
    handle: Texture_Handle
}

Depth_State :: struct
{
    mode: Depth_Flags,
    compare: Compare_Op
}

Blend_State :: struct
{
    enable: bool,
    color_op: Blend_Op,
    src_color_factor: Blend_Factor,
    dst_color_factor: Blend_Factor,
    alpha_op: Blend_Op,
    src_alpha_factor: Blend_Factor,
    dst_alpha_factor: Blend_Factor,
    color_write_mask: u8,
}

Draw_Indexed_Indirect_Command :: struct
{
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    vertex_offset: i32,
    first_instance: u32,
}

Dispatch_Indirect_Command :: struct
{
    num_groups_x: u32,
    num_groups_y: u32,
    num_groups_z: u32,
}

BVH_Instance :: struct
{
    transform: [12]f32,  // Row-major 3x4 matrix!
    using _: bit_field u32 {
        custom_idx: u32 | 24,
        mask:       u32 | 8,
    },
    using _: bit_field u32 {
        _unused: u32 | 24,
        disable_culling: bool | 1,
        flip_facing: bool | 1,
        force_opaque: bool | 1,
        force_not_opaque: bool | 1,
        force_opacity_micromaps: bool | 1,
        disable_opacity_micromaps: bool | 1,
        _unused_flags: bool | 2,
    },
    blas_root: rawptr,
}

BVH_Mesh_Desc :: struct
{
    opacity: BVH_Opacity,
    vertex_stride: u32,
    max_vertex: u32,  // e.g. if reading vertices [200..300], this value must be 300.
    tri_count: u32,
}
BVH_AABB_Desc :: struct
{
    opacity: BVH_Opacity,
    stride: u32,
    aabb_count: u32,
}
BVH_Shape_Desc :: union { BVH_Mesh_Desc, BVH_AABB_Desc }

BVH_Mesh  :: struct { verts: rawptr, indices: rawptr }
BVH_AABBs :: struct { data: rawptr }
BVH_Shape :: union { BVH_Mesh, BVH_AABBs }

BLAS_Desc :: struct
{
    hint: BVH_Hint,
    caps: BVH_Capabilities,
    shapes: []BVH_Shape_Desc,
}

TLAS_Desc :: struct
{
    hint: BVH_Hint,
    caps: BVH_Capabilities,
    instance_count: u32,
}

Device_Limits :: struct
{
    max_anisotropy: f32,
}

// Procedures

// Initialization and interaction with the OS.
init: proc(validation := true, loc := #caller_location) : _init
cleanup: proc(loc := #caller_location) : _cleanup
wait_idle: proc() : _wait_idle
swapchain_init: proc(surface: vk.SurfaceKHR, init_size: [2]u32, frames_in_flight: u32) : _swapchain_init
swapchain_resize: proc(size: [2]u32) : _swapchain_resize  // NOTE: Do not call this every frame! Only if the dimensions change.
swapchain_acquire_next: proc() -> Texture : _swapchain_acquire_next  // Blocks CPU until at least one frame is available.
swapchain_present: proc(queue: Queue, sem_wait: Semaphore, wait_value: u64) : _swapchain_present
features_available: proc() -> Features : _features_available
device_limits: proc() -> Device_Limits : _device_limits

// Memory
gpuptr :: struct { ptr: rawptr, _impl: [2]u64 }
ptr :: struct { cpu: rawptr, using gpu: gpuptr }
null :: gpuptr {}
mem_alloc_raw: proc(#any_int el_size, #any_int el_count, #any_int align: i64, mem_type := Memory.Default, alloc_type := Allocation_Type.Default, loc := #caller_location) -> ptr : _mem_alloc_raw
mem_suballoc: proc(addr: ptr, offset, el_size, el_count: i64, loc := #caller_location) -> ptr : _mem_suballoc
mem_free_raw: proc(addr: gpuptr, loc := #caller_location) : _mem_free_raw

// Textures
texture_size_and_align: proc(desc: Texture_Desc, loc := #caller_location) -> (size: u64, align: u64) : _texture_size_and_align
texture_create: proc(desc: Texture_Desc, storage: gpuptr, queue: Queue = nil, signal_sem: Semaphore = {}, signal_value: u64 = 0, name := "", loc := #caller_location) -> Texture : _texture_create
texture_destroy: proc(texture: Texture, loc := #caller_location) : _texture_destroy
texture_view_descriptor: proc(texture: Texture, view_desc: Texture_View_Desc, loc := #caller_location) -> Texture_Descriptor : _texture_view_descriptor
texture_rw_view_descriptor: proc(texture: Texture, view_desc: Texture_View_Desc, loc := #caller_location) -> Texture_Descriptor : _texture_rw_view_descriptor
sampler_descriptor: proc(sampler_desc: Sampler_Desc, loc := #caller_location) -> Sampler_Descriptor : _sampler_descriptor
texture_view_descriptor_size: proc() -> u32 : _texture_view_descriptor_size
texture_rw_view_descriptor_size: proc() -> u32 : _texture_rw_view_descriptor_size
sampler_descriptor_size: proc() -> u32 : _sampler_descriptor_size

// Shaders
shader_create: proc(code: []u32, type: Shader_Type_Graphics, entry_point_name := "main", name := "", loc := #caller_location) -> Shader : _shader_create
shader_create_compute: proc(code: []u32, group_size_x: u32, group_size_y: u32 = 1, group_size_z: u32 = 1, entry_point_name := "main", name := "", loc := #caller_location) -> Shader : _shader_create_compute
shader_destroy: proc(shader: Shader, loc := #caller_location) : _shader_destroy

// Semaphores
semaphore_create: proc(init_value: u64 = 0, name := "", loc := #caller_location) -> Semaphore : _semaphore_create
semaphore_wait: proc(sem: Semaphore, wait_value: u64, loc := #caller_location) : _semaphore_wait
semaphore_destroy: proc(sem: Semaphore, loc := #caller_location) : _semaphore_destroy

// Queues
queue_wait_idle: proc(queue: Queue) : _queue_wait_idle
queue_submit: proc(queue: Queue, cmd_bufs: []Command_Buffer, loc := #caller_location) : _queue_submit

// Raytracing
blas_size_and_align: proc(desc: BLAS_Desc, loc := #caller_location) -> (size: u64, align: u64) : _blas_size_and_align
blas_create: proc(desc: BLAS_Desc, storage: gpuptr, name := "", loc := #caller_location) -> BVH : _blas_create
blas_build_scratch_buffer_size_and_align: proc(desc: BLAS_Desc, loc := #caller_location) -> (size: u64, align: u64) : _blas_build_scratch_buffer_size_and_align
tlas_size_and_align: proc(desc: TLAS_Desc, loc := #caller_location) -> (size: u64, align: u64) : _tlas_size_and_align
tlas_create: proc(desc: TLAS_Desc, storage: gpuptr, name := "", loc := #caller_location) -> BVH : _tlas_create
tlas_build_scratch_buffer_size_and_align: proc(desc: TLAS_Desc, loc := #caller_location) -> (size: u64, align: u64) : _tlas_build_scratch_buffer_size_and_align
bvh_size_and_align :: proc { blas_size_and_align, tlas_size_and_align }
bvh_create :: proc { blas_create, tlas_create }
bvh_build_scratch_buffer_size_and_align :: proc { blas_build_scratch_buffer_size_and_align, tlas_build_scratch_buffer_size_and_align }
bvh_root_ptr: proc(bvh: BVH, loc := #caller_location) -> rawptr : _bvh_root_ptr
bvh_descriptor: proc(bvh: BVH, loc := #caller_location) -> BVH_Descriptor : _bvh_descriptor
bvh_descriptor_size: proc() -> u32 : _bvh_descriptor_size
bvh_destroy: proc(bvh: BVH, loc := #caller_location) : _bvh_destroy

// Command buffer
commands_begin: proc(queue: Queue, loc := #caller_location) -> Command_Buffer : _commands_begin

// Commands
cmd_mem_copy_raw: proc(cmd_buf: Command_Buffer, dst, src: gpuptr, #any_int bytes: i64, loc := #caller_location) : _cmd_mem_copy_raw
cmd_copy_to_texture: proc(cmd_buf: Command_Buffer, texture: Texture, src, dst: gpuptr, loc := #caller_location) : _cmd_copy_to_texture
cmd_copy_mips_to_texture: proc(cmd_buf: Command_Buffer, texture: Texture, src_buffer: gpuptr, regions: []Mip_Copy_Region, loc := #caller_location) : _cmd_copy_mips_to_texture
cmd_blit_texture: proc(cmd_buf: Command_Buffer, src, dst: Texture, src_rects: []Blit_Rect, dst_rects: []Blit_Rect, filter: Filter, loc := #caller_location) : _cmd_blit_texture

cmd_set_desc_heap: proc(cmd_buf: Command_Buffer, textures, textures_rw, samplers, bvhs: gpuptr, loc := #caller_location) : _cmd_set_desc_heap

cmd_add_wait_semaphore: proc(cmd_buf: Command_Buffer, sem: Semaphore, wait_value: u64, loc := #caller_location) : _cmd_add_wait_semaphore
cmd_add_signal_semaphore: proc(cmd_buf: Command_Buffer, sem: Semaphore, signal_value: u64, loc := #caller_location) : _cmd_add_signal_semaphore

cmd_barrier: proc(cmd_buf: Command_Buffer, before: Stage, after: Stage, hazards: Hazard_Flags = {}, loc := #caller_location) : _cmd_barrier

cmd_set_shaders: proc(cmd_buf: Command_Buffer, vert_shader: Shader, frag_shader: Shader, loc := #caller_location) : _cmd_set_shaders
cmd_set_compute_shader: proc(cmd_buf: Command_Buffer, compute_shader: Shader, loc := #caller_location) : _cmd_set_compute_shader
cmd_set_depth_state: proc(cmd_buf: Command_Buffer, state: Depth_State, loc := #caller_location) : _cmd_set_depth_state
cmd_set_blend_state: proc(cmd_buf: Command_Buffer, state: Blend_State, loc := #caller_location) : _cmd_set_blend_state

// Run compute shader based on number of groups
cmd_dispatch: proc(cmd_buf: Command_Buffer, compute_data: gpuptr, num_groups_x: u32, num_groups_y: u32 = 1, num_groups_z: u32 = 1, loc := #caller_location) : _cmd_dispatch

// Schedule indirect compute shader based on number of groups, arguments is a pointer to a Dispatch_Indirect_Command struct
cmd_dispatch_indirect: proc(cmd_buf: Command_Buffer, compute_data, arguments: gpuptr, loc := #caller_location) : _cmd_dispatch_indirect

cmd_begin_render_pass: proc(cmd_buf: Command_Buffer, desc: Render_Pass_Desc, loc := #caller_location) : _cmd_begin_render_pass
cmd_end_render_pass: proc(cmd_buf: Command_Buffer, loc := #caller_location) : _cmd_end_render_pass

// Indices, vertex_data and fragment_data can be nil
cmd_draw_indexed_instanced: proc(cmd_buf: Command_Buffer, vertex_data, fragment_data, indices: gpuptr,
                                 index_count: u32, instance_count: u32 = 1, loc := #caller_location) : _cmd_draw_indexed_instanced
cmd_draw_indexed_instanced_indirect: proc(cmd_buf: Command_Buffer, vertex_data, fragment_data, indices,
                                          indirect_arguments: gpuptr, loc := #caller_location) : _cmd_draw_indexed_instanced_indirect
cmd_draw_indexed_instanced_indirect_multi: proc(cmd_buf: Command_Buffer, vertex_data, fragment_data, indices: gpuptr,
                                                indirect_arguments: gpuptr, stride: u32, draw_count: gpuptr, loc := #caller_location) : _cmd_draw_indexed_instanced_indirect_multi

cmd_build_blas: proc(cmd_buf: Command_Buffer, bvh: BVH, bvh_storage, scratch_storage: gpuptr, shapes: []BVH_Shape, loc := #caller_location) : _cmd_build_blas
cmd_build_tlas: proc(cmd_buf: Command_Buffer, bvh: BVH, bvh_storage, scratch_storage: gpuptr, instances: gpuptr, loc := #caller_location) : _cmd_build_tlas

/////////////////////////
// Userland Utilities

// Memory

ptr_apply_offset :: #force_inline proc(addr: ^ptr, #any_int offset: i64)
{
    if addr.cpu != nil {
        addr.cpu = auto_cast(uintptr(addr.cpu) + uintptr(offset))
    }
    addr.gpu.ptr = auto_cast(uintptr(addr.gpu.ptr) + uintptr(offset))
}

ptr_t :: struct($T: typeid)
{
    cpu: ^T,
    using gpu: gpuptr,
}

slice_t :: struct($T: typeid)
{
    cpu: []T,
    using gpu: gpuptr,
}

mem_alloc_ptr :: #force_inline proc($T: typeid, mem_type := Memory.Default, loc := #caller_location) -> ptr_t(T)
{
    p := mem_alloc_raw(size_of(T), 1, align_of(T), mem_type = mem_type, loc = loc)
    return ptr_t(T) {
        cpu = cast(^T) p.cpu,
        gpu = p.gpu
    }
}

mem_alloc_slice :: #force_inline proc($T: typeid, #any_int count: i32, mem_type := Memory.Default, loc := #caller_location) -> slice_t(T)
{
    p := mem_alloc_raw(size_of(T), count, align_of(T), mem_type = mem_type, loc = loc)
    return slice_t(T) {
        cpu = slice.from_ptr(cast(^T)p.cpu, int(count)),
        gpu = p.gpu
    }
}

mem_alloc :: proc {
    mem_alloc_ptr,
    mem_alloc_slice,
}

mem_free_ptr :: #force_inline proc(addr: ptr_t($T))
{
    mem_free_raw(addr.gpu)
}

mem_free_slice :: #force_inline proc(addr: slice_t($T))
{
    mem_free_raw(addr.gpu)
}

mem_free :: proc {
    mem_free_ptr,
    mem_free_slice,
}

cmd_mem_copy_ptr :: #force_inline proc(cmd_buf: Command_Buffer, dst: ptr_t($T), src: ptr_t(T), loc := #caller_location)
{
    cmd_mem_copy_raw(cmd_buf, dst.gpu, src.gpu, size_of(T), loc = loc)
}

cmd_mem_copy_slice :: proc(cmd_buf: Command_Buffer, dst: slice_t($T), src: slice_t(T), #any_int count: i32, loc := #caller_location)
{
    cmd_mem_copy_raw(cmd_buf, dst.gpu, src.gpu, size_of(T) * count, loc = loc)
}

cmd_mem_copy :: proc {
    cmd_mem_copy_ptr,
    cmd_mem_copy_slice,
}

// Simple linear allocator. Not thread-safe, as it is meant for
// temporary, thread-local allocations (e.g. staging buffers).
Arena :: struct
{
    block_size: i64,
    mem_type: Memory,

    offset: i64,
    block_idx: i64,
    blocks: [dynamic]Arena_Block,
}

@(private="file")
Arena_Block :: struct
{
    p: ptr,
    size: i64,
}

arena_init :: proc(#any_int block_size: i64 = 4*1024*1024, mem_type := Memory.Default) -> Arena
{
    res: Arena
    res.block_size = block_size
    res.mem_type = mem_type
    first_block := Arena_Block {
        p = mem_alloc_raw(block_size, 1, 16, mem_type = mem_type),
        size = block_size,
    }
    append(&res.blocks, first_block)
    return res
}

arena_alloc_raw :: proc(arena: ^Arena, #any_int el_size: i64, #any_int el_count: i64, #any_int align: i32 = 16) -> ptr
{
    bytes := el_size * el_count
    assert(bytes > 0 && align > 0)

    block := arena.blocks[arena.block_idx]

    // If we request an alignment of > 16 and cpu/gpu are only aligned to 16,
    // it's impossible to find the same offset for both.
    if block.p.cpu != nil && uintptr(block.p.cpu) % uintptr(align) != uintptr(block.p.gpu.ptr) % uintptr(align) {
        panic("Could not satisfy alignment requirements in GPU arena allocation.")
    }

    gpu_addr := uintptr(block.p.gpu.ptr) + uintptr(arena.offset)
    arena.offset = i64(align_up(u64(gpu_addr), u64(align)) - u64(uintptr(block.p.gpu.ptr)))
    if arena.offset + bytes > block.size {
        block = arena_next_block(arena, bytes)
        arena.offset = 0
    }

    suballoc_ptr := mem_suballoc(block.p, arena.offset, el_size, el_count)
    arena.offset += bytes

    return suballoc_ptr

    align_up :: proc(x, align: u64) -> (aligned: u64)
    {
        assert(0 == (align & (align - 1)), "must align to a power of two")
        return (x + (align - 1)) &~ (align - 1)
    }

    arena_next_block :: proc(arena: ^Arena, bytes: i64) -> Arena_Block
    {
        arena.block_idx += 1
        arena.offset = 0
        if arena.block_idx >= i64(len(arena.blocks))
        {
            new_size := max(arena.block_size, bytes)
            new_p := mem_alloc_raw(new_size, 1, 16, mem_type = arena.mem_type)
            new_block := Arena_Block { p = new_p, size = new_size }
            append(&arena.blocks, new_block)
            return new_block
        }
        else
        {
            if arena.blocks[arena.block_idx].size <= bytes
            {
                return arena.blocks[arena.block_idx]
            }
            else
            {
                mem_free_raw(arena.blocks[arena.block_idx].p.gpu)
                new_size := max(arena.block_size, bytes)
                new_p := mem_alloc_raw(new_size, 1, 16, mem_type = arena.mem_type)
                new_block := Arena_Block { p = new_p, size = new_size }
                arena.blocks[arena.block_idx] = new_block
                return new_block
            }
        }
    }
}

arena_alloc_ptr :: #force_inline proc(arena: ^Arena, $T: typeid) -> ptr_t(T)
{
    return transmute(ptr_t(T)) arena_alloc_raw(arena, size_of(T), 1, align_of(T))
}

arena_alloc_slice :: #force_inline proc(arena: ^Arena, $T: typeid, #any_int count: i32) -> slice_t(T)
{
    p_raw := arena_alloc_raw(arena, size_of(T), count, align_of(T))
    return slice_t(T) {
        cpu = slice.from_ptr(cast(^T) p_raw.cpu, int(count)),
        gpu = p_raw.gpu
    }
}

arena_alloc :: proc {
    arena_alloc_ptr,
    arena_alloc_slice,
}

arena_free_all :: proc(arena: ^Arena)
{
    arena.offset = 0
    arena.block_idx = 0
}

arena_destroy :: proc(arena: ^Arena)
{
    for block in arena.blocks {
        mem_free_raw(block.p.gpu)
    }
    delete(arena.blocks)
    arena^ = {}
}

Owned_Texture :: struct
{
    using tex: Texture,
    mem: gpuptr,
}

texture_alloc_and_create :: proc(desc: Texture_Desc, queue: Queue = nil, signal_sem: Semaphore = {}, signal_value: u64 = 0, name := "", loc := #caller_location) -> Owned_Texture
{
    size, align := texture_size_and_align(desc)
    ptr := mem_alloc_raw(size, 1, align, .GPU, .Texture, loc = loc)
    texture := texture_create(desc, ptr, queue, signal_sem, signal_value, name = name, loc = loc)
    return Owned_Texture { texture, ptr.gpu }
}

texture_free_and_destroy :: proc(texture: ^Owned_Texture)
{
    texture_destroy(texture)
    mem_free_raw(texture.mem)
    texture^ = {}
}

set_texture_desc :: #force_inline proc(desc_heap: ptr, idx: u32, desc: Texture_Descriptor)
{
    desc_size := #force_inline texture_view_descriptor_size()
    tmp := desc
    runtime.mem_copy(auto_cast(uintptr(desc_heap.cpu) + uintptr(idx * desc_size)), &tmp, int(desc_size))
}

set_texture_rw_desc :: #force_inline proc(desc_heap: ptr, idx: u32, desc: Texture_Descriptor)
{
    desc_size := #force_inline texture_rw_view_descriptor_size()
    tmp := desc
    runtime.mem_copy(auto_cast(uintptr(desc_heap.cpu) + uintptr(idx * desc_size)), &tmp, int(desc_size))
}

set_sampler_desc :: #force_inline proc(desc_heap: ptr, idx: u32, desc: Sampler_Descriptor)
{
    desc_size := #force_inline sampler_descriptor_size()
    tmp := desc
    runtime.mem_copy(auto_cast(uintptr(desc_heap.cpu) + uintptr(idx * desc_size)), &tmp, int(desc_size))
}

set_bvh_desc :: #force_inline proc(desc_heap: ptr, idx: u32, desc: BVH_Descriptor)
{
    desc_size := #force_inline bvh_descriptor_size()
    tmp := desc
    runtime.mem_copy(auto_cast(uintptr(desc_heap.cpu) + uintptr(idx * desc_size)), &tmp, int(desc_size))
}

Owned_BVH :: struct
{
    using handle: BVH,
    mem: gpuptr,
}

blas_alloc_and_create :: proc(desc: BLAS_Desc) -> Owned_BVH
{
    size, align := bvh_size_and_align(desc)
    ptr := mem_alloc_raw(size, 1, align, .GPU)
    bvh := bvh_create(desc, ptr)
    return Owned_BVH { bvh, ptr }
}

tlas_alloc_and_create :: proc(desc: TLAS_Desc) -> Owned_BVH
{
    size, align := bvh_size_and_align(desc)
    ptr := mem_alloc_raw(size, 1, align, .GPU)
    bvh := bvh_create(desc, ptr)
    return Owned_BVH { bvh, ptr }
}

bvh_alloc_and_create :: proc { blas_alloc_and_create, tlas_alloc_and_create }

bvh_free_and_destroy :: proc(bvh: ^Owned_BVH)
{
    bvh_destroy(bvh)
    mem_free_raw(bvh.mem)
    bvh^ = {}
}

blas_alloc_build_scratch_buffer :: proc(arena: ^Arena, desc: BLAS_Desc) -> ptr
{
    size, align := blas_build_scratch_buffer_size_and_align(desc)
    return arena_alloc_raw(arena, size, 1, align)
}

tlas_alloc_build_scratch_buffer :: proc(arena: ^Arena, desc: TLAS_Desc) -> ptr
{
    size, align := tlas_build_scratch_buffer_size_and_align(desc)
    return arena_alloc_raw(arena, size, 1, align)
}

bvh_alloc_build_scratch_buffer :: proc { blas_alloc_build_scratch_buffer, tlas_alloc_build_scratch_buffer }

// Swapchain utils

swapchain_init_from_sdl :: proc(window: ^sdl.Window, frames_in_flight: u32)
{
    vk_surface: vk.SurfaceKHR
    ok := sdl.Vulkan_CreateSurface(window, get_vulkan_instance(), nil, &vk_surface)
    ensure(ok, "Could not create surface.")

    window_size_x: i32
    window_size_y: i32
    sdl.GetWindowSize(window, &window_size_x, &window_size_y)
    swapchain_init(vk_surface, { u32(max(0, window_size_x)), u32(max(0, window_size_y)) }, frames_in_flight)
}

// Texture utils

cmd_generate_mipmaps :: proc(cmd_buf: Command_Buffer, texture: Texture)
{
    for mip in 1..<texture.mip_count
    {
        if mip > 1 {
            cmd_barrier(cmd_buf, .Transfer, .Transfer)
        }

        src := Blit_Rect { mip_level = mip - 1 }
        dst := Blit_Rect { mip_level = mip }
        cmd_blit_texture(cmd_buf, texture, texture, { src }, { dst }, .Linear)
    }
}
