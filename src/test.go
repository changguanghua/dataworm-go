package main

import (
	"fmt"
)

func main(){
	q := make(chan int, 10)
	
	q <- 1
	
	for {
		if len(q) == 0{
			break
		}
		a := <-q
		fmt.Println(a)
		if a < 10{
			q <- a * 2
		}
	}
	close(q)
}