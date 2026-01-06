#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_scalar_block_layout : require

layout(location = 0) out vec4 _res_out_loc0_;

layout(buffer_reference) readonly buffer _res_ptr_void;
layout(buffer_reference) readonly buffer _res_slice_Vertex;
layout(buffer_reference) readonly buffer _res_ptr_Data;

struct Vertex
{
    vec4 pos;
    vec4 color;
};

struct Data
{
    _res_slice_Vertex verts;
};

struct Output
{
    vec4 pos;
    vec4 color;
};

void main();
layout(buffer_reference, scalar) readonly buffer _res_ptr_void { uint _res_void_; };
layout(buffer_reference, scalar) readonly buffer _res_slice_Vertex { Vertex _res_[]; };
layout(buffer_reference, scalar) readonly buffer _res_ptr_Data { Data _res_; };

layout(set = 0, binding = 0) uniform texture2D _res_textures_[];
layout(set = 1, binding = 0) uniform writeonly image2D _res_textures_rw_[];
layout(set = 2, binding = 0) uniform sampler _res_samplers_[];

layout(push_constant, scalar) uniform Push
{
    _res_ptr_Data _res_vert_data_;
    _res_ptr_Data _res_frag_data_;
    _res_ptr_void _res_vert_indirect_data_;
    _res_ptr_void _res_frag_indirect_data_;
};

void main()
{
    uint vert_id = gl_VertexIndex;
    _res_ptr_Data data = _res_vert_data_;
    Output vert_out;
    vert_out.pos = vec4(data._res_.verts._res_[vert_id].pos.xyz, 1.0);
    vert_out.color = data._res_.verts._res_[vert_id].color;
    gl_Position = vert_out.pos; _res_out_loc0_ = vert_out.color; ;
}

