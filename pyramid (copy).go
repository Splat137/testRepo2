package kata

func LongestSlideDown(pyramid [][]int) int {
	if pyramid == nil || len(pyramid) == 0 {
		return 0
	}
	if len(pyramid) == 1 {
		return pyramid[0][0]
	}

	max := 0

	for i := len(pyramid) - 2; i > 0; i-- {
		for j := 0; j < len(pyramid[i]); j++ {
			max = 0
			if pyramid[i+1][j] > pyramid[i+1][j+1] {
				max = pyramid[i+1][j]
			}
			if pyramid[i+1][j] <= pyramid[i+1][j+1] {
				max = pyramid[i+1][j+1]
			}fdsfdsf
			pyramid[i][j] += max
		}fsdzvbjyuxtgfyjcftjmtutyuj
		
		
		;.oli;.niol;
	}

	if pyramid[1][0] > pyramid[1][1] {
		return pyramid[1][0] + pyramid[0][0]
	}
	if pyramid[1][1] >= pyramid[1][0] {
		return pyramid[1][1] + pyramid[0][0]
	}

	return 0
}
