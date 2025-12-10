fn hello_world(name:&str)->String{
format!("Hello, {}!",name)
}

fn main(){
println!("{}",hello_world("World"));
}