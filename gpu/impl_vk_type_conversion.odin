
#+private
package gpu

import vk "vendor:vulkan"
import "core:sync"
import "base:runtime"

to_vk_shader_stage :: #force_inline proc(type: Shader_Type_Graphics) -> vk.ShaderStageFlags
{
    switch type
    {
        case .Vertex: return { .VERTEX }
        case .Fragment: return { .FRAGMENT }
    }
    return {}
}

to_vk_stage :: #force_inline proc(stage: Stage) -> vk.PipelineStageFlags
{
    switch stage
    {
        case .Transfer: return { .TRANSFER }
        case .Compute: return { .COMPUTE_SHADER }
        case .Raster_Color_Out: return { .COLOR_ATTACHMENT_OUTPUT }
        case .Fragment_Shader: return { .FRAGMENT_SHADER }
        case .Vertex_Shader: return { .VERTEX_SHADER }
        case .Build_BVH: return { .ACCELERATION_STRUCTURE_BUILD_KHR }
        case .All: return { .ALL_COMMANDS }
    }
    return {}
}

to_vk_load_op :: #force_inline proc(load_op: Load_Op) -> vk.AttachmentLoadOp
{
    switch load_op
    {
        case .Clear: return .CLEAR
        case .Load: return .LOAD
        case .Dont_Care: return .DONT_CARE
    }
    return {}
}

to_vk_store_op :: #force_inline proc(store_op: Store_Op) -> vk.AttachmentStoreOp
{
    switch store_op
    {
        case .Store: return .STORE
        case .Dont_Care: return .DONT_CARE
    }
    return {}
}

to_vk_compare_op :: #force_inline proc(compare_op: Compare_Op) -> vk.CompareOp
{
    switch compare_op
    {
        case .Never: return .NEVER
        case .Less: return .LESS
        case .Equal: return .EQUAL
        case .Less_Equal: return .LESS_OR_EQUAL
        case .Greater: return .GREATER
        case .Not_Equal: return .NOT_EQUAL
        case .Greater_Equal: return .GREATER_OR_EQUAL
        case .Always: return .ALWAYS
    }
    return {}
}

to_vk_texture_type :: #force_inline proc(type: Texture_Type) -> vk.ImageType
{
    switch type
    {
        case .D2: return .D2
        case .D3: return .D3
        case .D1: return .D1
    }
    return {}
}

to_vk_texture_view_type :: #force_inline proc(type: Texture_Type) -> vk.ImageViewType
{
    switch type
    {
        case .D2: return .D2
        case .D3: return .D3
        case .D1: return .D1
    }
    return {}
}

to_vk_texture_format :: proc(format: Texture_Format) -> vk.Format
{
    switch format
    {
        case .Default: panic("Implementation bug!")
        case .RGBA8_Unorm: return .R8G8B8A8_UNORM
        case .BGRA8_Unorm: return .B8G8R8A8_UNORM
        case .RGBA8_SRGB: return .R8G8B8A8_SRGB
        case .D32_Float: return .D32_SFLOAT
        case .RGBA16_Float: return .R16G16B16A16_SFLOAT
        case .RGBA32_Float: return .R32G32B32A32_SFLOAT
        case .BC1_RGBA_Unorm: return .BC1_RGBA_UNORM_BLOCK
        case .BC3_RGBA_Unorm: return .BC3_UNORM_BLOCK
        case .BC7_RGBA_Unorm: return .BC7_UNORM_BLOCK
        case .ASTC_4x4_RGBA_Unorm: return .ASTC_4x4_UNORM_BLOCK
        case .ETC2_RGB8_Unorm: return .ETC2_R8G8B8_UNORM_BLOCK
        case .ETC2_RGBA8_Unorm: return .ETC2_R8G8B8A8_UNORM_BLOCK
        case .EAC_R11_Unorm: return .EAC_R11_UNORM_BLOCK
        case .EAC_RG11_Unorm: return .EAC_R11G11_UNORM_BLOCK
    }
    return {}
}

to_vk_sample_count :: proc(sample_count: u32) -> vk.SampleCountFlags
{
    switch sample_count
    {
        case 0: return { ._1 }
        case 1: return { ._1 }
        case 2: return { ._2 }
        case 4: return { ._4 }
        case 8: return { ._8 }
        case: panic("Unsupported sample count.")
    }
    return {}
}

to_vk_texture_usage :: proc(usage: Usage_Flags) -> vk.ImageUsageFlags
{
    res: vk.ImageUsageFlags
    if .Sampled in usage do                  res += { .SAMPLED }
    if .Storage in usage do                  res += { .STORAGE }
    if .Color_Attachment in usage do         res += { .COLOR_ATTACHMENT }
    if .Depth_Stencil_Attachment in usage do res += { .DEPTH_STENCIL_ATTACHMENT }
    if .Transfer_Src in usage do             res += { .TRANSFER_SRC }
    return res
}

to_vk_filter :: proc(filter: Filter) -> vk.Filter
{
    switch filter
    {
        case .Linear: return .LINEAR
        case .Nearest: return .NEAREST
    }
    return {}
}

to_vk_mipmap_filter :: proc(filter: Filter) -> vk.SamplerMipmapMode
{
    switch filter
    {
        case .Linear: return .LINEAR
        case .Nearest: return .NEAREST
    }
    return {}
}

to_vk_address_mode :: proc(addr_mode: Address_Mode) -> vk.SamplerAddressMode
{
    switch addr_mode
    {
        case .Repeat: return .REPEAT
        case .Mirrored_Repeat: return .MIRRORED_REPEAT
        case .Clamp_To_Edge: return .CLAMP_TO_EDGE
    }
    return {}
}

to_vk_blas_desc :: proc(blas_desc: BLAS_Desc, arena: runtime.Allocator) -> vk.AccelerationStructureBuildGeometryInfoKHR
{
    geometries := make([]vk.AccelerationStructureGeometryKHR, len(blas_desc.shapes), allocator = arena)
    for &geom, i in geometries
    {
        switch shape in blas_desc.shapes[i]
        {
            case BVH_Mesh_Desc:
            {
                flags: vk.GeometryFlagsKHR = { .OPAQUE } if shape.opacity == .Fully_Opaque else {}
                geom = vk.AccelerationStructureGeometryKHR {
                    sType = .ACCELERATION_STRUCTURE_GEOMETRY_KHR,
                    flags = flags,
                    geometryType = .TRIANGLES,
                    geometry = { triangles = {
                        sType = .ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
                        vertexFormat = .R32G32B32_SFLOAT,
                        vertexData = {},
                        vertexStride = vk.DeviceSize(shape.vertex_stride),
                        maxVertex = shape.max_vertex,
                        indexType = .UINT32,
                        indexData = {},
                        transformData = {},
                    } }
                }
            }
            case BVH_AABB_Desc:
            {
                flags: vk.GeometryFlagsKHR = { .OPAQUE } if shape.opacity == .Fully_Opaque else {}
                geom = vk.AccelerationStructureGeometryKHR {
                    sType = .ACCELERATION_STRUCTURE_GEOMETRY_KHR,
                    flags = flags,
                    geometryType = .AABBS,
                    geometry = { aabbs = {
                        sType = .ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR,
                        stride = vk.DeviceSize(shape.stride),
                        data = {},
                    } }
                }
            }
        }
    }

    return vk.AccelerationStructureBuildGeometryInfoKHR {
        sType = .ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
        flags = to_vk_bvh_flags(blas_desc.hint, blas_desc.caps),
        type = .BOTTOM_LEVEL,
        mode = .BUILD,
        geometryCount = u32(len(geometries)),
        pGeometries = raw_data(geometries)
    }
}

to_vk_tlas_desc :: proc(tlas_desc: TLAS_Desc, arena: runtime.Allocator) -> vk.AccelerationStructureBuildGeometryInfoKHR
{
    geometry := new(vk.AccelerationStructureGeometryKHR)
    geometry^ = {
        sType = .ACCELERATION_STRUCTURE_GEOMETRY_KHR,
        geometryType = .INSTANCES,
        geometry = {
            instances = {
                sType = .ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR,
                arrayOfPointers = false,
                data = {
                    // deviceAddress = vku.get_buffer_device_address(device, instances_buf)
                }
            }
        }
    }

    return vk.AccelerationStructureBuildGeometryInfoKHR {
        sType = .ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
        flags = to_vk_bvh_flags(tlas_desc.hint, tlas_desc.caps),
        type = .TOP_LEVEL,
        mode = .BUILD,
        geometryCount = 1,
        pGeometries = geometry
    }
}

to_vk_bvh_flags :: proc(hint: BVH_Hint, caps: BVH_Capabilities) -> vk.BuildAccelerationStructureFlagsKHR
{
    flags: vk.BuildAccelerationStructureFlagsKHR
    if .Update in caps do            flags += { .ALLOW_UPDATE }
    if .Compaction in caps do        flags += { .ALLOW_COMPACTION }
    if hint == .Prefer_Fast_Trace do flags += { .PREFER_FAST_TRACE }
    if hint == .Prefer_Fast_Build do flags += { .PREFER_FAST_BUILD }
    if hint == .Prefer_Low_Memory do flags += { .LOW_MEMORY }

    return flags
}

to_vk_blend_op :: proc(op: Blend_Op) -> vk.BlendOp
{
    switch op
    {
        case .Add:          return .ADD
        case .Subtract:     return .SUBTRACT
        case .Rev_Subtract: return .REVERSE_SUBTRACT
        case .Min:          return .MIN
        case .Max:          return .MAX
    }
    return {}
}

to_vk_blend_factor :: proc(blend: Blend_Factor) -> vk.BlendFactor
{
    switch blend
    {
        case .Zero:      return .ZERO
        case .One:       return .ONE
        case .Src_Color: return .SRC_COLOR
        case .Dst_Color: return .DST_COLOR
        case .Src_Alpha: return .SRC_ALPHA
    }
    return {}
}

