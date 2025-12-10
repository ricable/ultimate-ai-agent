package main

import "fmt"

func helloWorld(name string) string {
	return fmt.Sprintf("Hello, %s!", name)
}

func main() {
	fmt.Println(helloWorld("World"))
}
