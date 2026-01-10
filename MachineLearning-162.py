import numpy as np
import statistics as st 
import random 

#1st question
def countletters(string):
    vowels = "aeiouAEIOU"
    consonants = "bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ"
    vowel = 0
    consonant = 0
    for char in string:
        if char in vowels:
            vowel += 1
        elif char in consonants:
            consonant += 1
    return (vowel, consonant)

#2nd question 
def multiply(a,b):

   if a.shape[1]!=b.Shape[0]:
    return f"Error: Cannot multiply matrices with shapes"
   else:
      c = np.dot(a,b)
      return c
   
#3rd question
def commomelements(list1,list2):
   count=0
   templist = list2.copy()
   for x in list1:
      if x in templist:
          count+=1
          templist.remove(x)
   return count

#4th question

def matrixtranspose(matrix):
   return matrix.T 

#5th question
def statistics(list):
   return f"Mean: {st.mean(list)}, Median: {st.median(list)}, Mode: {st.mode(list)}"

#1st Question
print("1st Question\nEnter the String: ")
string = str(input())
print(countletters(string))

#2nd Question
print("\n2nd Question")
rows = int(input('enter rows of matrix A:'))
columns = int(input("enter columns of matrix A:"))
print("elements of matrix A:")

A = np.array([list(map(int, input().split())) for _ in range(rows)])

rowsofb = int(input('enter rows of matrix B:'))
columnsofb = int(input("enter columns of matrix B:"))
print("elements of matrix B:")

B = np.array([list(map(int, input().split())) for _ in range(rowsofb)])
print(multiply(A,B))

#3rd Question
print("\n3rd Question")
list1 = [1, 2, 3, 4, 5, 1, 2]
list2 = [4, 5, 1, 6, 7, 1, 2]
print(commomelements(list1, list2))

#4th question
print("\n4th Question")
i = int(input("Enter number of rows: "))
print("Enter the elements rowwise:")
matrix = np.array([list(map(int, input().split())) for _ in range(i)])
print("Transposed Matrix:")
print(matrixtranspose(matrix))

#5th question
print("\n5th Question")
numbers = []
for i in range(100):
    numbers.append(random.randint(1, 50))
print(statistics(numbers))