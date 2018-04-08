# IngeniousHackathon_Talisman

project idea domain: Data Science
title: CV Recommendation Model
Won Grand Prize in this project in the 36-hour hackathon.

Input: Candidate's CV, Target field of the candidate
Output: Best sub-Job domain for the candidate, Recommendation of required skills for the target field

Tools used: Python, Rstudio

Description: First we developed a baseline model that suggest missing skills in candidate by using our ideal skill-set data. To recommend sub-domain, we developed a mathematical model which gives weights to skills by incorporating skill experience information. And give scores to a different domain. (for that candidate) We recommend highest score domain to the candidate.

Developed hybrid model, which is able to do both things. Also as new data of candidate comes our model automatically update its skill set to capture recent trends in each domain. As data increases, our recommendation becomes data-driven.

Applied Natural Language Processing to process project description written in CV. After applying NLP, we use statistical analysis of skills. Then we recommend job subdomain for that candidate. This inference is useful for skill recommendation module for other candidates.

Extension of the project: Developing candidate recommendation system for organization.

Team Members:
  Charmi Chokshi,
  Divya Dass,
  Parth Gadoya
