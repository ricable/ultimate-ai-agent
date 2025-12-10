package main

import "testing"

func TestGreet(t *testing.T) {
	result := Greet("World")
	expected := "Hello, World!"
	if result != expected {
		t.Errorf("Greet('World') = %s; want %s", result, expected)
	}
}

func TestAdd(t *testing.T) {
	result := Add(2, 3)
	expected := 5
	if result != expected {
		t.Errorf("Add(2, 3) = %d; want %d", result, expected)
	}
}
