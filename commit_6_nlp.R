install.packages('tm')
install.packages('SnowballC')
library(tm)
library(SnowballC)
install.packages("stringr", dependencies = TRUE)
library(stringr)

Clean_String <- function(string){
  # Lowercase
  temp <- tolower(string)
  # Remove everything that is not a number or letter (may want to keep more 
  # stuff in your actual analyses). 
  temp <- stringr::str_replace_all(temp,"[^a-zA-Z\\s]", " ")
  # Shrink down to just one white space
  temp <- stringr::str_replace_all(temp,"[\\s]+", " ")
  # Split it
  temp <- stringr::str_split(temp, " ")[[1]]
  # Get rid of trailing "" if necessary
  indexes <- which(temp == "")
  if(length(indexes) > 0){
    temp <- temp[-indexes]
  } 
  return(temp)
}

#project definitions
sentence <- "Predicted salary of an employee based on a set of Attributes 
such as years of experience, level of education, department, previous job salary, 
etc. using Univariate and Multivariate Linear Regression in R language and got 
90% prediction power."

sentence <- "implemented Convolution Neural Network to classify Pencil sketch vs Color sketch of 
#human face. Model got 100% accuracy in python language."

sentence <- "Implementation of PCA from scratch for dimensionality reduction of input images, 
LDA for reducing computation time required for calculation of within-class and inter-class 
scatter matrix and KNN classifier is done in Python. The proposed 
algorithm is tested on three different datasets and gives 100% accuracy on two of them."

sentence <- "1. Collected dataset of person's height(meter) and corresponding weight(Kilograms).
2. Implemented Polynomial Regression technique from scratch to build a predictive model 
for predicting  the weight of a person using his/her height dataset. (in Cpp)
3. Applied several degrees of a polynomials(quadratic, cubic etc) and 
estimated sum of squared error to measure efficiency
4. Used most efficient predictive model for predicting weight on unseen/new height dataset"

#clean sentence
clean_sentence <- Clean_String(sentence)

#remove stop words
stopwords_regex = paste(stopwords('en'), collapse = '\\b|\\b')
stopwords_regex = paste0('\\b', stopwords_regex, '\\b')
documents = stringr::str_replace_all(clean_sentence, stopwords_regex, ' ')
#remove blank spaces
a=documents[documents != " "]
#remove repitive characters
s=unique(a[a != ""])

setwd("C:/Users/HP/Desktop")
d = read.csv("hack.csv")

temp<-vector()

k=1

for (j in 1:length(s))
{
    for (i in 1:nrow(d))
    {
      if(s[j] == d$Concepts[i])
      {
        print(s[j])
        print(d$Concepts[i])
        temp[k]=s[j]
        k=k+1
      }
  }
}
print("Found Data Science Concepts used in given definition:")
print(temp)
