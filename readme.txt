
Compilation:
	g++ -fopenmp -O3 -march=native -std=c++11 painterly_rendering.cpp -o painterly_rendering `pkg-config --cflags --libs opencv`

Usage:
	./painterly_rendering <Image File> <Style: 0~3> <num_threads>
	style 0: Impressionist
	style 1: Expressionist
	Style 2: Colorist Wash
	Style 3: Pointillist 

Output:
	We save our output as final_<style>.jpg