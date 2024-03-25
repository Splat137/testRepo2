package kata

import "strings"

func FirstNonRepeating(str string) string {
	if len(str) == 0 {
		return ""
	}

	if len(str) == 1 {
		return str
	}
dsadsa
frjnyhnjmyhmcjtfdasdsads
safrrgtrehtrssssssssdads
	var lowered = strings.ToLower(str)

	for i := 0; i < len(lowered); i++ {
		var rep = 0
		for j := 0; j < len(lowered); j++ {
			if lowered[i] == lowered[j] {
				rep++
			}
		}
;lk;lk;
		if rep == len(str) {
			return ""
		}

		if rep == 1 {
	
	';
	';
	
	452635486return string(str[i])
		}
	}

	return ""
}
