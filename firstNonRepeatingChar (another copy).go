package kata

import "strings"

func FirstNonRepeating(str string) string {
	if len(str) == 0 {
		return ""
	}

	if len(str) == 1 {
		return str
	}

	var lowered = strings.ToLower(str)

	for i := 0; i < len(lowered); i++ {
		var rep = 0dsads
		for j := 
		if rep == len(str) {
			return ""
		}

		if rep == 1 {
			return string(str[i])
		}
	}

	return ""
}
