
#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require

layout(location = 0) out vec3 _res_out_loc0_;

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
    vec3 color;
};

layout(buffer_reference) readonly buffer _res_ptr_void { uint _res_void_; };
layout(buffer_reference) readonly buffer _res_slice_Vertex { Vertex _res_; };
layout(buffer_reference) readonly buffer _res_ptr_Data { Data _res_; };

layout(push_constant, std140) uniform Push
{
    _res_ptr_Data _res_data_;
    _res_ptr_Data _res_frag_data_;
};

void main()
{
    uint vert_id = gl_VertexIndex;
    _res_ptr_Data data = _res_data_;
    Output vert_out;
    vert_out.pos = data.verts[vert_id].pos;
    vert_out.color = vec3(1.0, 0.0, 0.0);
    gl_Position = vert_out.pos; _res_out_loc0_ = vert_out.color; ;
}

