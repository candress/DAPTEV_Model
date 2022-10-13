import random


def get_rand_seq(min = 10, max = 30, gc_percent = 0.5):		
	length = random.randint(min, max)
	sequence = ""

	for i in range(length):
		choice = random.random()   
		if choice < gc_percent:
			base = "g" if random.random() < 0.5 else "c"
		else:
			base = "a" if random.random() < 0.5 else "u"

		sequence += base

	return sequence


def get_seq(len = 30, gc_percent = 0.5):
	return get_rand_seq(len,len,gc_percent)