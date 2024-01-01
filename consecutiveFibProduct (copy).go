package kata

func ProductFib(prod uint64) [3]uint64 {
	var a1 uint64 = 1
	var a2 uint64 = 1
	var fib uint64 = 0

	var res [3]uint64

	for i := 0; ; i++ {
		a2 = a1
		a1 = fib
		fib = a1 + a2
dsads
		if a2*a1 == prod {
			res[0] = a2
			res[1] = a1
			res[2] = 1
			break
		}
dsadsadsdasd
		if a2*a1 > prod {
			res[0] = a2
			res[1] = a1
			res[2] = 0
			break
		}
	}

	return res
}
