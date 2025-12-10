//! AST types for CUDA to Rust transpilation

use std::fmt;

/// For loop initialization type
#[derive(Debug, Clone, PartialEq)]
pub enum ForInit {
    /// Variable declaration: int i = 0
    VarDecl { name: String, ty: String, init: Box<Expr> },
    /// Expression: i = 0
    Expr(Box<Expr>),
}

/// Expression type
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Literal(String),
    Identifier(String),
    Binary { op: String, left: Box<Expr>, right: Box<Expr> },
    Unary { op: String, expr: Box<Expr> },
    Call { name: String, args: Vec<Expr> },
    Index { expr: Box<Expr>, index: Box<Expr> },
    Member { expr: Box<Expr>, member: String },
}

/// Statement type
#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    Expression(Expr),
    Assignment { lhs: Expr, rhs: Expr },
    VarDecl { name: String, ty: String, init: Option<Expr> },
    If { cond: Expr, then_stmt: Box<Stmt>, else_stmt: Option<Box<Stmt>> },
    For { init: ForInit, cond: Expr, update: Expr, body: Box<Stmt> },
    While { cond: Expr, body: Box<Stmt> },
    Block(Vec<Stmt>),
    Return(Option<Expr>),
    Break,
    Continue,
}

/// Function definition
#[derive(Debug, Clone, PartialEq)]
pub struct Function {
    pub name: String,
    pub params: Vec<(String, String)>,
    pub return_type: String,
    pub body: Vec<Stmt>,
    pub is_kernel: bool,
}

/// AST root
#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub functions: Vec<Function>,
    pub globals: Vec<Stmt>,
}

impl fmt::Display for ForInit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ForInit::VarDecl { name, ty, init } => write!(f, "{ty} {name} = {init:?}"),
            ForInit::Expr(expr) => write!(f, "{expr:?}"),
        }
    }
}