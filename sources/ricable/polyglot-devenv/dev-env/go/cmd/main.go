package main

import (
	"fmt"
	"log"
)

// Greet returns a greeting message
func Greet(name string) string {
	return fmt.Sprintf("Hello, %s!", name)
}

// Add performs addition of two integers
func Add(a, b int) int {
	return a + b
}

func main() {
	log.Println("Go Development Environment")
	fmt.Println(Greet("World"))
	fmt.Printf("2 + 3 = %d\n", Add(2, 3))
}
