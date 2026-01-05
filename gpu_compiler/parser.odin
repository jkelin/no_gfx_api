
#+feature dynamic-literals

package main

import "base:runtime"
import "core:slice"
import intr "base:intrinsics"

Any_Node :: union
{
    Any_Statement,
    Any_Expr,
    ^Ast_Decl,
}

Ast :: struct
{
    used_types: [dynamic]Ast_Type,
    used_in_locations: map[u32]Ast_Type,
    used_out_locations: map[u32]Ast_Type,
    used_data_type: ^Ast_Type,
    used_indirect_data_type: ^Ast_Type,
    scope: ^Ast_Scope,
    procs: [dynamic]^Ast_Proc_Def,
}

Ast_Node :: struct
{
    token: Token,
    derived: Any_Node,
}

Ast_Scope :: struct
{
    enclosing_scope: ^Ast_Scope,  // Can be nil
    decls: [dynamic]^Ast_Decl,
}

// Declarations (global)

Ast_Decl :: struct
{
    using base: Ast_Node,
    name: string,
    type: ^Ast_Type,
    attr: Maybe(Ast_Attribute),
}

Ast_Proc_Def :: struct
{
    decl: ^Ast_Decl,
    statements: []^Ast_Statement,
    scope: ^Ast_Scope,
}

// Expressions

Any_Expr :: union
{
    ^Ast_Binary_Expr,
    ^Ast_Member_Access,
    ^Ast_Array_Access,
    ^Ast_Ident_Expr,
    ^Ast_Lit_Expr,
    ^Ast_Call,
}

Ast_Expr :: struct
{
    using base: Ast_Node,
    derived_expr: Any_Expr,
    type: ^Ast_Type,
}

Ast_Attribute_Type :: enum
{
    Vert_ID,
    Position,
    Data,
    Instance_ID,
    Draw_ID,
    Indirect_Data,

    // With args:
    Out_Loc,
    In_Loc,
}

Ast_Attribute :: struct
{
    type: Ast_Attribute_Type,
    arg: u32
}

Ast_Binary_Op :: enum
{
    Add,
    Minus,
    Mul,
    Div,
}

Ast_Binary_Expr :: struct
{
    using base_expr: Ast_Expr,
    lhs: ^Ast_Expr,
    rhs: ^Ast_Expr,
    op: Ast_Binary_Op,
}

Ast_Member_Access :: struct
{
    using base_expr: Ast_Expr,
    target: ^Ast_Expr,
    member_name: string,
}

Ast_Array_Access :: struct
{
    using base_expr: Ast_Expr,
    target: ^Ast_Expr,
    idx_expr: ^Ast_Expr,
}

Ast_Call :: struct
{
    using base_expr: Ast_Expr,
    target: ^Ast_Expr,
    args: []^Ast_Expr,
}

Ast_Ident_Expr :: struct
{
    using base_expr: Ast_Expr
}

Ast_Lit_Expr :: struct
{
    using base_expr: Ast_Expr
}

// Statements

Any_Statement :: union
{
    ^Ast_Assign,
    ^Ast_Stmt_Expr,
    ^Ast_Return,
}

Ast_Statement :: struct
{
    using base: Ast_Node,
    derived_statement: Any_Statement,
}

Ast_Assign :: struct
{
    using base_statement: Ast_Statement,
    lhs: ^Ast_Expr,
    rhs: ^Ast_Expr
}

Ast_Stmt_Expr :: struct
{
    using base_statement: Ast_Statement,
    expr: ^Ast_Expr,
}

Ast_Return :: struct
{
    using base_statement: Ast_Statement,
    expr: ^Ast_Expr,
}

// Types

Ast_Type_Kind :: enum
{
    Poison = 0,
    Label,
    Pointer,
    Slice,
    Proc,
    Primitive,
    Struct,
}

Ast_Type_Primitive_Kind :: enum
{
    None = 0,
    Float,
    Uint,
    Int,
    Texture_ID,
    Sampler_ID,
    Vec2,
    Vec3,
    Vec4,
    Mat4,
}

Ast_Type :: struct
{
    kind: Ast_Type_Kind,
    primitive_kind: Ast_Type_Primitive_Kind,  // Only populated if kind == .Primitive
    base: ^Ast_Type,

    name: Token,

    // Proc
    args: []^Ast_Decl,
    ret: ^Ast_Type,
    ret_attr: Maybe(Ast_Attribute),

    // Struct
    members: []^Ast_Decl,
}

parse_file :: proc(filename: string, tokens: []Token, allocator: runtime.Allocator) -> (Ast, bool)
{
    context.allocator = allocator

    parser := Parser {
        tokens = tokens,
        filename = filename,
    }
    ast := _parse_file(&parser)
    return ast, !parser.error
}

Parser :: struct
{
    tokens: []Token,
    filename: string,
    at: u32,
    error: bool,
    scope: ^Ast_Scope,
    used_types: [dynamic]Ast_Type,
    used_out_locations: map[u32]Ast_Type,
    used_in_locations: map[u32]Ast_Type,
    used_data_type: ^Ast_Type,
    used_indirect_data_type: ^Ast_Type,
}

_parse_file :: proc(using p: ^Parser) -> Ast
{
    ast := Ast {
        scope = new(Ast_Scope)
    }

    scope = ast.scope

    loop: for true
    {
        #partial switch tokens[at].type
        {
            case .Ident:
            {
                if tokens[at+1].type == .Colon &&
                   tokens[at+2].type == .Colon &&
                   tokens[at+3].type == .Struct
                {
                    parse_struct_def(p)
                }
                else if tokens[at+1].type == .Colon &&
                        tokens[at+2].type == .Colon &&
                        tokens[at+3].type == .LParen
                {
                    append(&ast.procs, parse_proc_def(p))
                }
                else
                {
                    parse_error(p, "Expecting ':: (' or ':: struct' after the identifier")
                    break loop
                }
            }
            case .EOS: break loop
            case:
            {
                parse_error(p, "Expecting an identifier at top level.")
                break loop
            }
        }
    }

    ast.used_types = used_types
    ast.used_out_locations = used_out_locations
    ast.used_in_locations = used_in_locations
    ast.used_data_type = used_data_type
    ast.used_indirect_data_type = used_indirect_data_type
    return ast
}

parse_struct_def :: proc(using p: ^Parser) -> ^Ast_Decl
{
    node := make_node(p, Ast_Decl)
    append(&scope.decls, node)

    struct_type := new(Ast_Type)
    struct_type.kind = .Struct
    node.type = struct_type

    ident := required_token(p, .Ident)
    node.name = ident.text

    required_token(p, .Colon)
    required_token(p, .Colon)
    required_token(p, .Struct)
    required_token(p, .LBrace)
    struct_type.members = parse_decl_list(p, false)
    required_token(p, .RBrace)
    return node
}

parse_proc_def :: proc(using p: ^Parser) -> ^Ast_Proc_Def
{
    decl := make_node(p, Ast_Decl)
    append(&scope.decls, decl)

    proc_type := new(Ast_Type)
    proc_type.kind = .Proc
    decl.type = proc_type

    old_scope := scope
    scope = new(Ast_Scope)
    scope.enclosing_scope = old_scope
    defer scope = old_scope

    proc_def := new(Ast_Proc_Def)
    proc_def.decl = decl
    proc_def.scope = scope

    ident := required_token(p, .Ident)
    decl.name = ident.text

    required_token(p, .Colon)
    required_token(p, .Colon)
    required_token(p, .LParen)
    proc_type.args = parse_decl_list(p, true)
    required_token(p, .RParen)

    if optional_token(p, .Arrow)
    {
        proc_type.ret = parse_type(p)
        if tokens[at].type == .Attribute {
            proc_type.ret_attr = parse_attribute(p)
            if proc_type.ret_attr != nil
            {
                if proc_type.ret_attr.?.type == .In_Loc {
                    used_in_locations[proc_type.ret_attr.?.arg] = proc_type.ret^
                } else if proc_type.ret_attr.?.type == .Out_Loc {
                    used_out_locations[proc_type.ret_attr.?.arg] = proc_type.ret^
                }
            }
        }
    }

    required_token(p, .LBrace)
    proc_def.statements = parse_statement_list(p)
    required_token(p, .RBrace)
    return proc_def
}

parse_statement_list :: proc(using p: ^Parser) -> []^Ast_Statement
{
    scratch, _ := acquire_scratch()
    tmp_list := make([dynamic]^Ast_Statement, allocator = scratch)
    for true
    {
        stmt := parse_statement(p)
        if stmt != nil do append(&tmp_list, stmt)
        if tokens[at].type == .RBrace || tokens[at].type == .EOS do break
        if error do break
    }
    return slice.clone(tmp_list[:])
}

parse_statement :: proc(using p: ^Parser) -> ^Ast_Statement
{
    for optional_token(p, .Semi) {}

    cursor := at
    for ; tokens[cursor].type != .Semi && tokens[cursor].type != .Assign && tokens[cursor].type != .EOS; cursor += 1 { }

    found_assign := tokens[cursor].type == .Assign
    node: ^Ast_Statement
    if found_assign
    {
        node = parse_assign(p)
    }
    else if optional_token(p, .Return)
    {
        ret_stmt := make_statement(p, Ast_Return)
        ret_stmt.expr = parse_expr(p)
        node = ret_stmt
    }
    else if tokens[at].type == .Ident && tokens[at+1].type == .Colon
    {
        parse_var_decl(p)
    }
    else
    {
        stmt_expr := make_statement(p, Ast_Stmt_Expr)
        stmt_expr.expr = parse_expr(p)
        node = stmt_expr
    }

    required_token(p, .Semi)
    return node
}

parse_assign :: proc(using p: ^Parser) -> ^Ast_Statement
{
    node := make_statement(p, Ast_Assign)
    node.lhs = parse_expr(p)
    required_token(p, .Assign)
    node.rhs = parse_expr(p)
    return node
}

parse_var_decl :: proc(using p: ^Parser)
{
    node := make_node(p, Ast_Decl)
    append(&scope.decls, node)

    ident := required_token(p, .Ident)
    node.name = ident.text

    required_token(p, .Colon)
    node.type = parse_type(p)
}

parse_expr :: proc(using p: ^Parser, prec: int = max(int)) -> ^Ast_Expr
{
    lhs: ^Ast_Expr

    // Prefix operators

    // Postfix operators
    lhs = parse_postfix_expr(p)

    // Binary operators
    for true
    {
        op, found := Op_Precedence[tokens[at].type]
        undo_recurse := false
        undo_recurse |= !found
        undo_recurse |= op.prec > prec  // If it's less important (=greater priority) don't recurse.
        if undo_recurse do return lhs

        // Recurse
        bin_op := make_expr(p, Ast_Binary_Expr)
        bin_op.op = op.op
        bin_op.lhs = lhs
        at += 1
        bin_op.rhs = parse_expr(p)
        lhs = bin_op
    }

    return lhs
}

parse_primary_expr :: proc(using p: ^Parser) -> ^Ast_Expr
{
    expr: ^Ast_Expr
    if optional_token(p, .LParen)
    {
        internal := parse_expr(p)
        required_token(p, .RParen)
        expr = internal
    }
    else if tokens[at].type == .Ident
    {
        expr = make_expr(p, Ast_Ident_Expr)
        at += 1
    }
    else if tokens[at].type == .NumLit
    {
        expr = make_expr(p, Ast_Lit_Expr)
        at += 1
    }

    return expr
}

parse_postfix_expr :: proc(using p: ^Parser) -> ^Ast_Expr
{
    expr := parse_primary_expr(p)
    loop: for true
    {
        #partial switch tokens[at].type
        {
            case .Dot:
            {
                member_access := make_expr(p, Ast_Member_Access)
                at += 1

                ident := required_token(p, .Ident)
                member_access.member_name = ident.text
                member_access.target = expr
                expr = member_access
            }
            case .LParen:
            {
                call := make_expr(p, Ast_Call)
                call.target = expr
                at += 1

                if tokens[at].type != .RParen
                {
                    scratch, _ := acquire_scratch()
                    tmp_list := make([dynamic]^Ast_Expr, allocator = scratch)
                    for true
                    {
                        append(&tmp_list, parse_expr(p))
                        comma_present := optional_token(p, .Comma)
                        if !comma_present do break
                        if comma_present && (tokens[at].type == .RParen || tokens[at].type == .RBrace) do break
                        if error do break
                    }

                    call.args = slice.clone(tmp_list[:])
                }

                required_token(p, .RParen)

                expr = call
            }
            case .LBracket:
            {
                array_access := make_expr(p, Ast_Array_Access)
                at += 1

                array_access.idx_expr = parse_expr(p)
                array_access.target = expr
                expr = array_access

                required_token(p, .RBracket)
            }
            case: break loop
        }
    }

    return expr
}

parse_decl_list :: proc(using p: ^Parser, add_to_scope: bool) -> []^Ast_Decl
{
    scratch, _ := acquire_scratch()
    tmp_list := make([dynamic]^Ast_Decl, allocator = scratch)
    for true
    {
        append(&tmp_list, parse_decl_list_elem(p, add_to_scope))
        comma_present := optional_token(p, .Comma)
        if !comma_present do break
        if comma_present && (tokens[at].type == .RParen || tokens[at].type == .RBrace) do break
        if error do break
    }
    return slice.clone(tmp_list[:])
}

parse_decl_list_elem :: proc(using p: ^Parser, add_to_scope: bool) -> ^Ast_Decl
{
    node := make_node(p, Ast_Decl)
    if add_to_scope {
        append(&scope.decls, node)
    }

    ident := required_token(p, .Ident)
    required_token(p, .Colon)

    node.name = ident.text

    node.type = parse_type(p)
    node.attr = parse_attribute(p)
    if node.attr != nil
    {
        if node.attr.?.type == .Data {
            used_data_type = node.type
        } else if node.attr.?.type == .Indirect_Data {
            used_indirect_data_type = node.type
        }

        if node.attr.?.type == .In_Loc {
            used_in_locations[node.attr.?.arg] = node.type^
        } else if node.attr.?.type == .Out_Loc {
            used_out_locations[node.attr.?.arg] = node.type^
        }
    }

    return node
}

parse_type :: proc(using p: ^Parser) -> ^Ast_Type
{
    base: ^Ast_Type
    node: ^Ast_Type

    for true
    {
        if optional_token(p, .LBracket)
        {
            slice_type := new(Ast_Type)
            slice_type.kind = .Slice
            if node != nil do node.base = slice_type
            node = slice_type
            if base == nil do base = node

            required_token(p, .RBracket)
        }
        else if optional_token(p, .Caret)
        {
            ptr_type := new(Ast_Type)
            ptr_type.kind = .Pointer
            if node != nil do node.base = ptr_type
            node = ptr_type
            if base == nil do base = node
        }
        else do break
    }

    ident := required_token(p, .Ident)
    prim_type: Ast_Type_Primitive_Kind
    switch ident.text
    {
        case "float": prim_type = .Float
        case "uint": prim_type = .Uint
        case "int": prim_type = .Int
        case "vec2": prim_type = .Vec2
        case "vec3": prim_type = .Vec3
        case "vec4": prim_type = .Vec4
        case "textureid": prim_type = .Texture_ID
        case "samplerid": prim_type = .Sampler_ID
        case "mat4": prim_type = .Mat4
        case: prim_type = .None
    }

    ident_node := new(Ast_Type)
    ident_node.name = ident
    ident_node.primitive_kind = prim_type
    if node != nil do node.base = ident_node
    node = ident_node
    if base == nil do base = node

    if prim_type == .None {
        node.kind = .Label
    } else {
        node.kind = .Primitive
    }

    add_type_if_not_present(p, base)
    return base
}

parse_attribute :: proc(using p: ^Parser) -> Maybe(Ast_Attribute)
{
    if tokens[at].type != .Attribute do return nil

    attr := Ast_Attribute {}

    token := required_token(p, .Attribute)

    switch token.text
    {
        case "vert_id": attr.type = .Vert_ID
        case "position": attr.type = .Position
        case "data": attr.type = .Data
        case "instance_id": attr.type = .Instance_ID
        case "draw_id": attr.type = .Draw_ID
        case "indirect_data": attr.type = .Indirect_Data
        case "in_loc":
        {
            // ??? Why is the compiler making me do this?
            attr.type, _ = .In_Loc,
            required_token(p, .LParen)
            num_token := required_token(p, .NumLit)
            val, ok := num_token.value.(u64)
            if !ok do parse_error_on_token(p, num_token, "Expecting integer value on attribute arguments.")
            attr.arg = u32(val)
            required_token(p, .RParen)
        }
        case "out_loc":
        {
            // ??? Why is the compiler making me do this?
            attr.type, _ = .Out_Loc,
            required_token(p, .LParen)
            num_token := required_token(p, .NumLit)
            val, ok := num_token.value.(u64)
            if !ok do parse_error_on_token(p, num_token, "Expecting integer value on attribute arguments.")
            attr.arg = u32(val)
            required_token(p, .RParen)
        }
        case:
        {
            parse_error(p, "Unknown attribute '%v'.", token.text)
        }
    }

    return attr
}

make_node :: proc(using p: ^Parser, $T: typeid) -> ^T
{
    node := new(T)
    node.token = tokens[at]
    return node
}

make_expr :: proc(using p: ^Parser, $T: typeid) -> ^T
{
    node := new(T)
    node.token = tokens[at]
    node.derived_expr = node
    return node
}

make_statement :: proc(using p: ^Parser, $T: typeid) -> ^T
{
    node := new(T)
    node.token = tokens[at]
    node.derived_statement = node
    return node
}

parse_error :: proc(using p: ^Parser, fmt_str: string, args: ..any)
{
    parse_error_on_token(p, tokens[at], fmt_str, ..args)
}

parse_error_on_token :: proc(using p: ^Parser, token: Token, fmt_str: string, args: ..any)
{
    if error do return

    error_msg(filename, token, fmt_str, ..args)
    error = true
}

required_token :: proc(using p: ^Parser, type: Token_Type) -> Token
{
    if tokens[at].type != type
    {
        parse_error(p, "Unexpected token '%v': expecting '%v'", tokens[at].text, type)
        return {}
    }

    at += 1
    return tokens[at-1]
}

optional_token :: proc(using p: ^Parser, type: Token_Type) -> bool
{
    if tokens[at].type == type
    {
        at += 1
        return true
    }

    return false
}

// Operator precedence
Op_Info :: struct
{
    prec: int,
    op: Ast_Binary_Op,
}
Op_Precedence := map[Token_Type]Op_Info {
    .Mul = { 3, .Mul },
    .Div = { 3, .Div },

    .Plus  = { 4, .Add },
    .Minus = { 4, .Minus },
}

add_type_if_not_present :: proc(using p: ^Parser, type: ^Ast_Type)
{
    if type == nil do return

    for &used in used_types {
        if same_type(type, &used) do return
    }

    append(&used_types, type^)
}
