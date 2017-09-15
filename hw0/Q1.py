import sys

inFilePath = (sys.argv)[1]
def find_word(my_list, word):
	ret_value = False
	index = -1
	for i in range(len(my_list)):
		if(my_list[i][0] == word):
			ret_value = True
			index = i
			break
	return (ret_value, index)

inF = open(inFilePath,'r')
wordList = (inF.read()).split()

wordCount = []
for i in range(len(wordList)):
	word = wordList[i]
	query = find_word(wordCount,word)
	if(query[0]==True):
		wordCount[query[1]] = (wordCount[query[1]][0],wordCount[query[1]][1]+1)
	else:
		wordCount.append((word,1))

outF = open('Q1.txt','w')
for i in range(len(wordCount)):
	line = wordCount[i][0]+" "+str(i)+" "+str(wordCount[i][1])
	if(i!=len(wordCount)-1):
		line += '\n'
	outF.write(line)
inF.close()
outF.close()
