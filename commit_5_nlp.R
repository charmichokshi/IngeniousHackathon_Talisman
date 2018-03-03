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

sentence <- "Predicted salary of an employee based on a set of Attributes 
such as years of experience, level of education, department, previous job salary, 
etc. using Univariate and Multivariate Linear Regression in R language and got 
90% prediction power."


#implemented Convolution Neural Network to classify Pencil sketch vs Color image of 
#human face. Model got 100% accuracy in python language.


clean_sentence <- Clean_String(sentence)
clean_sentence


stopwords_regex = paste(stopwords('en'), collapse = '\\b|\\b')
stopwords_regex = paste0('\\b', stopwords_regex, '\\b')
documents = stringr::str_replace_all(clean_sentence, stopwords_regex, ' ')
documents

a=gsub(" ", "", documents)
a

a=a[a != ""]
a

s=unique(a[a != ""])
s
s[1]
