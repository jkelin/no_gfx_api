
package gpu

import "impl"

// This API follow ZII (Zero Is Initialization) principles. Initializing to 0
// will yield predictable and reasonable behavior in general.

// Handles
Handle :: rawptr
Texture :: distinct Handle
Command_Buffer :: distinct Handle
Semaphore :: distinct Handle

// Enums
Memory :: enum { Default = 0, GPU, Readback }
Texture_Type :: enum { D2 = 0, D3, D1, Cube, D2_Array, Cube_Array }
Texture_Format :: enum { None = 0, Rgba8_Unorm, D32_Float, Rg11B10_Float, Rgb10_A2_Unorm }
Usage :: enum { Sampled = 0, Storage, Color_Attachment, Depth_Stencil_Attachment }
Usage_Flags :: bit_set[Usage; u32]

// Constants
All_Mips: u8 : max(u8)
All_Layers: u16 : max(u16)

// Structs
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

Texture_View_Desc :: struct
{
    format: Texture_Format,
    base_mip: u32,
    mip_count: u8,     // 0 = All_Mips
    base_layer: u16,
    layer_count: u16,  // 0 = All_Layers
}

// Initialization. This is simpler than it would actually be, for brevity.
init: proc(window: ^sdl.Window) : impl.init

// Memory
malloc :: proc(bytes: u64, align: u64 = 1, mem := Memory.Default) -> rawptr { return {} }
free :: proc(ptr: rawptr) {}
host_to_device_ptr :: proc(ptr: rawptr) {}

// Textures
texture_size_and_align :: proc(desc: Texture_Desc) -> (size: u64, align: u64) { return 0, 0 }
texture_view_descriptor :: proc(texture: Texture, view_desc: Texture_View_Desc) -> [4]u64 { return {} }
texture_rw_view_descriptor :: proc(texture: Texture, view_desc: Texture_View_Desc) -> [4]u64 { return {} }

// Semaphores
sem_create :: proc(init_value: u64) -> Semaphore { return {} }

// Commands
cmd_mem_copy :: proc(cmd_buf: Command_Buffer, src, dst: rawptr)
cmd_copy_to_texture :: proc(cmd_buf: Command_Buffer, texture: Texture, src, dst: rawptr)

cmd_set_active_texture_heap_ptr :: proc(cmd_buf: Command_Buffer, ptr: rawptr)

cmd_barrier :: proc() {}
cmd_signal_after :: proc() {}
cmd_wait_before :: proc() {}

cmd_set_pipeline :: proc() {}
cmd_set_depth_stencil_state :: proc() {}
cmd_set_blend_state :: proc() {}

cmd_dispatch :: proc() {}
cmd_dispatch_indirect :: proc() {}

cmd_begin_render_pass :: proc() {}
cmd_end_render_pass :: proc() {}

cmd_draw_indexed_instanced :: proc(cmd_buf: Command_Buffer, vertex_data: rawptr, pixel_data: rawptr,
                                   indices: rawptr, index_count: u32, instance_count: u32) {}
