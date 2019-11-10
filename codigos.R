##########################################
## Curso de Introdução à Ciência de Dados
## Prof. Dr. Neylson Crepalde
##########################################

#######################
# Regressão Linear ####
#######################

# Se os pacotes necessários não estiverem instalados, faça a instalação
if (!require(tidyverse)) install.packages("tidyverse")
if (!require(ISLR)) install.packages("ISLR")
if (!require(texreg)) install.packages("texreg")
if (!require(splines)) install.packages("splines")
if (!require(gam)) install.packages("gam")
if (!require(tree)) install.packages("tree")
if (!require(randomForest)) install.packages("randomForest")

# Carrega os pacotes necessários
library(tidyverse)
library(ISLR)
library(texreg)
library(gam)
library(tree)
library(randomForest)

# Carrega os dados Advertisement
adv = read_csv("http://faculty.marshall.usc.edu/gareth-james/ISL/Advertising.csv") %>% 
  select(-X1)

adv

# Vamos fazer uma regressão para cada tipo de investimento
reg1 = lm(sales ~ TV, data = adv)
reg2 = lm(sales ~ radio, data = adv)
reg3 = lm(sales ~ newspaper, data = adv)

# Agora vamos fazer uma regressão com todos os preditores juntos
reg_completa = lm(sales ~ TV + radio + newspaper, data = adv)

# Visualizando os resultados dos modelos
summary(reg1)
summary(reg2)
summary(reg3)
summary(reg_completa)

# Uma visualização bacana de todos os modelos juntos
screenreg(list(reg1, reg2, reg3, reg_completa))

# Produz gráficos de avaliação dos resíduos
par(mfrow = c(2,2)) # Divide a tela de plotagem em 2 linhas e 2 colunas
plot(reg_completa)
par(mfrow = c(1,1)) # Volta a tela de plotagem para exibir 1 gráficos por vez


# Pega o intervalo de confiança para os valores preditos
fitted_confint = predict(reg1, interval = "prediction")

ggplot(adv, aes(x = TV)) +
  geom_point(aes(y = sales)) +
  geom_line(aes(y = fitted_confint[,1]), col = "red", lwd = 1) +
  geom_line(aes(y = fitted_confint[,2]), col = "blue", lwd = 1) +
  geom_line(aes(y = fitted_confint[,3]), col = "blue", lwd = 1)

# Predições
head(adv)
novosdados = tibble(TV=50, radio=10, newspaper=15)
predict(reg_completa, newdata = novosdados)


###########################
# Métodos não-lineares ####
###########################

#### 1. Regressão polinomial ####
# Vamos investigar a relação entre idade e salário. Primeira coisa,
# vamos ver como essas duas variáveis interagem na PNAD contínua 2019/2
data(Wage)
summary(Wage$age)
medias = sapply(min(Wage$age):max(Wage$age), 
                function(x) mean(Wage$wage[Wage$age == x], na.rm=T))

ggplot(NULL, aes(x=min(Wage$age):max(Wage$age), 
                 y=medias)) +
  geom_point(size=2) +
  labs(x='Age', y='Wage')


# Vamos verificar um ajuste para uma regressão linear simples e
# regressões polinomiais de grau 2 até 4
reg = lm(wage ~ age, data = Wage)
summary(reg)

# Vamos verificar o ajuste
age.grid = min(Wage$age):max(Wage$age)
preds = predict(reg,
                newdata=list(age=age.grid),
                se=TRUE)
# Calcula o intervalo de confiança
se.bands=cbind(preds$fit + 2*preds$se.fit,
               preds$fit - 2*preds$se.fit)


plot(Wage$age, Wage$wage, cex=.5,col="darkgrey")
lines(age.grid, preds$fit, lwd=2, col="blue")
matlines(age.grid,se.bands,lwd=1,col="blue",lty=3) # plota colunas de uma matrix

# Agora vamos tentar com um modelo polinomial de grau 2
reg2 = lm(wage ~ poly(age,2), data = Wage)
summary(reg2)

# Vamos verificar o ajuste
preds2 = predict(reg2,
                 newdata=list(age=age.grid),
                 se=TRUE)
# Calcula o intervalo de confiança
se.bands2 = cbind(preds2$fit + 2*preds2$se.fit,
                  preds2$fit - 2*preds2$se.fit)


plot(Wage$age, Wage$wage, cex=.5,col="darkgrey")
lines(age.grid, preds2$fit, lwd=2, col="blue")
matlines(age.grid, se.bands2, lwd=1, col="blue", lty=3) # plota colunas de uma matrix

# Quanto maior o polinômio, melhor o ajuste. Mas como saber quantos
# polinômios devemos usar para obter um bom ajuste?

# Testando a quantidade de polinômios necessários
fit.1=lm(wage~age,data=Wage)
fit.2=lm(wage~poly(age,2),data=Wage)
fit.3=lm(wage~poly(age,3),data=Wage)
fit.4=lm(wage~poly(age,4),data=Wage)
fit.5=lm(wage~poly(age,5),data=Wage)

# Checa com ANOVA
anova(fit.1,fit.2,fit.3,fit.4,fit.5)
# De acordo com a ANOVA, um polinômio de grau 3 é o suficiente pois
# não há melhora estatisticamente significativa com mais um polinômio.
preds3 = predict(fit.3,
                 newdata=list(age=age.grid),
                 se=TRUE)
# Calcula o intervalo de confiança
se.bands3 = cbind(preds3$fit + 2*preds3$se.fit,
                  preds3$fit - 2*preds3$se.fit)


plot(Wage$age, Wage$wage, cex=.5,col="darkgrey")
#title("Degree-4 Polynomial")
lines(age.grid, preds3$fit, lwd=2, col="blue")
matlines(age.grid, se.bands3, lwd=1, col="blue", lty=3) # plota colunas de uma matrix

# 2. Smoothing Splines
plot(Wage$age, Wage$wage, cex=.5, col="darkgrey")
title("Smoothing Spline")

fit  = smooth.spline(Wage$age, Wage$wage, df=16)
# Busca o melhor valor por Cross Validation
fit2 = smooth.spline(Wage$age, Wage$wage, cv=TRUE)
fit2$df

lines(fit, col="red",lwd=2)
lines(fit2, col="blue",lwd=2)
legend("topright",legend=c("16 DF","6.8 DF"),col=c("red","blue"),lty=1,lwd=2,cex=.8)

# Calcula o R2
r2_score = function(y_true, y_pred) {
  return(cor(y_true, y_pred)^2)
}
rmse = function(y_true, y_pred) {
  return( sqrt(mean((y_pred - y_true)^2)) )
}

summary(fit.3)$r.squared

r2_score(Wage$wage, predict(fit.3))
r2_score(Wage$wage, predict(fit2, Wage$age)$y)

rmse(Wage$wage, predict(fit.3))
rmse(Wage$wage, predict(fit2, Wage$age)$y)
