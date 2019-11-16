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
if (!require(titanic)) install.packages("titanic")
if (!require(pROC)) install.packages("pROC")
if (!require(texreg)) install.packages("texreg")
if (!require(splines)) install.packages("splines")
if (!require(gam)) install.packages("gam")
if (!require(tree)) install.packages("tree")
if (!require(randomForest)) install.packages("randomForest")

# Carrega os pacotes necessários
library(MASS)
library(tidyverse)
library(ISLR)
library(titanic)
library(pROC)
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
hist(resid(reg_completa))
par(mfrow = c(2,2)) # Divide a tela de plotagem em 2 linhas e 2 colunas
plot(reg_completa)
par(mfrow = c(1,1)) # Volta a tela de plotagem para exibir 1 gráficos por vez
plot(adv$sales, predict(reg_completa))

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

# A Abordagem de treino e teste
set.seed(123)
train = sample(nrow(adv), nrow(adv)*.7)
reg_completa = lm(sales ~ TV + radio + newspaper, data = adv[train,])

# Testando o erro
# Funções para calcular o R2 e o RMSE
r2_score = function(y_true, y_pred) {
  return(cor(y_true, y_pred)^2)
}
rmse = function(y_true, y_pred) {
  return( sqrt(mean((y_pred - y_true)^2)) )
}


# RMSE de treino
yhat_treino = predict(reg_completa)
rmse(adv$sales[train], yhat_treino)

# RMSE de teste
yhat_teste = predict(reg_completa, newdata = adv[-train, ])
rmse(adv$sales[-train], yhat_teste)


##########################
# Regressão Logística ####
##########################
#

data("titanic_train")
tt = titanic_train %>% as_tibble()
tt


# Investigando as chances de sobrevivência no titanic
set.seed(123)
train = sample(nrow(tt), nrow(tt)*.6)

logit = glm(Survived ~ Age + Sex + factor(Pclass), data = tt[train,],
            family = binomial())

summary(logit)

# Testando o ajuste
# Confusion matrix
yhat_train = round(predict(logit, tt[train,], type='response'))
table(tt$Survived[train], yhat_train)

# erro de teste
yhat_test = round(predict(logit, tt[-train,], type='response'))
table(tt$Survived[-train], yhat_test)

roc(tt$Survived[train], predict(logit, tt[train,], type='response'), 
    plot = T, main='Train ROC')

roc(tt$Survived[-train], predict(logit, tt[-train,], type='response'), 
    plot = T, main='Test ROC')


# Prevendo a minha probabilidade de sobrevivência no Titanic
eu = tibble(Age=32, Sex='male', Pclass=3)
predict(logit, newdata = eu, type='response')


###########################
# Métodos não-lineares ####
###########################

#### 1. Regressão polinomial ####
# Vamos investigar a relação entre idade e salário. Primeira coisa,
# vamos ver como essas duas variáveis interagem na PNAD contínua 2019/2
data(Wage)
summary(Wage$age)
summary(Wage$wage)
hist(Wage$age)
hist(Wage$wage)

plot(Wage$age, Wage$wage, cex=.5,col="darkgrey")
# Temos documentação!
?Wage

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

# 2. Smoothing Splines ####
set.seed(1)
train = sample(nrow(Wage), nrow(Wage)*.6)

fit  = smooth.spline(Wage$age[train], Wage$wage[train], df=16)
# Busca o melhor valor por Cross Validation
fit2 = smooth.spline(Wage$age[train], Wage$wage[train], cv=TRUE)
fit2$df


# Plotando no teste
plot(Wage$age[-train], Wage$wage[-train], cex=.5, col="darkgrey")
title("Smoothing Spline - Test Data")
lines(fit, col="red",lwd=2)
lines(fit2, col="blue",lwd=2)
legend("topright",legend=c("16 DF","6.7 DF"),col=c("red","blue"),lty=1,lwd=2,cex=.8)


# Calcula o R2
# R2 para treino
r2_score(Wage$wage[train], predict(fit, Wage$age[train])$y)
r2_score(Wage$wage[train], predict(fit2, Wage$age[train])$y)
# R2 para o teste
r2_score(Wage$wage[-train], predict(fit, Wage$age[-train])$y)
r2_score(Wage$wage[-train], predict(fit2, Wage$age[-train])$y)

# RMSE de treino
rmse(Wage$wage[train], predict(fit, Wage$age[train])$y)
rmse(Wage$wage[train], predict(fit2, Wage$age[train])$y)
# RMSE de teste
rmse(Wage$wage[-train], predict(fit, Wage$age[-train])$y)
rmse(Wage$wage[-train], predict(fit2, Wage$age[-train])$y)



# 3. GAM ####
## Para executar um GAM usando Natural splines, podemos utilizar
## o comando nativo de lm
gam1 = lm(wage ~ ns(year,4) + ns(age,5) + education, data=Wage)
summary(gam1)

## Para executar um GAM utilizando smoothing splines
## (o que não é trivial pois não faz OSL), a função
## gam() do pacote GAM implementa esse método usando
## uma abordagem conhecida como backfitting
## De acordo com os autores do livro ISLR
## "This method fits a model involving multiple predictors by
## repeatedly updating the fit for each predictor in turn,
## holding the others fixed.
set.seed(123)
train = sample(nrow(Wage), nrow(Wage)*.6)

gam.m1 = gam(wage ~ s(age,5) + education, data=Wage[train,])
gam.m2 = gam(wage ~ year + s(age,5) + education, data=Wage[train,])
gam.m3 = gam(wage ~ s(year,4) + s(age,5) + education, data=Wage[train,])
anova(gam.m1,gam.m2,gam.m3,test="F") # Teste de variância

# Pelo teste de variância o melhor modelo foi o segundo.

summary(gam.m2)

# Verificando os resultados dos modelos em gráficos
par(mfrow=c(1,3))
plot.Gam(gam1, se=TRUE, col="red")

par(mfrow=c(1,2))
plot(gam.m1, se=TRUE, col='orange')
par(mfrow=c(1,3))
plot(gam.m2, se=TRUE, col='orange')
plot(gam.m3, se=TRUE,col="blue")
par(mfrow=c(1,1))

# Verificando o ajuste
# Modelo 1, teste
r21 = r2_score(Wage$wage[-train], predict(gam.m1, newdata=Wage[-train,]))
rmse1 = rmse(Wage$wage[-train], predict(gam.m1, newdata=Wage[-train,]))

# Modelo 2, teste
r22 = r2_score(Wage$wage[-train], predict(gam.m2, newdata=Wage[-train,]))
rmse2 = rmse(Wage$wage[-train], predict(gam.m2, newdata=Wage[-train,]))

# Modelo 3, teste
r23 = r2_score(Wage$wage[-train], predict(gam.m3, newdata=Wage[-train,]))
rmse3 = rmse(Wage$wage[-train], predict(gam.m3, newdata=Wage[-train,]))

# Verifica os resultados no gráfico
plot(1:3, c(r21,r22,r23), type = 'b', lty='dashed', main="R2")
plot(1:3, c(rmse1, rmse2, rmse3), type = 'b', lty='dashed', main="RMSE")

#################################
# Métodos baseados em árvore ####
#################################

# 1. Decision trees para problemas de classificação ####
set.seed(123)
train = sample(nrow(tt), nrow(tt)*.6)

#tt$Pclass = as.factor(tt$Pclass)
tree.tt = tree(Survived ~ Age + Sex + Pclass, tt[train,])

summary(tree.tt)
plot(tree.tt)
text(tree.tt, pretty = 1)

yhat_tree = predict(tree.tt, tt[-train,])
roc(tt$Survived[-train], yhat_tree, plot=T)

plot(tt$Age, tt$Pclass*as.numeric(as.factor(tt$Sex)),
     col=alpha(tt$Survived+1, .4), pch=19)

tt %>% 
  ggplot(aes(x=Age, 
             y=Pclass*as.numeric(as.factor(Sex)),
             color=as.factor(Survived))) +
  geom_point() +
  scale_color_manual(values=c(adjustcolor('black', .4), 
                              adjustcolor('red', .4)),
                     name = 'Survived')

# 2. Decision trees para problemas de regressão ####
# Variável resposta
set.seed(1)
train = sample(nrow(Wage), nrow(Wage)*.6)

tree.wage = tree(wage ~ .-logwage, Wage[-train,])
summary(tree.wage)
plot(tree.wage)
text(tree.wage, pretty=0)

yhat_test = predict(tree.wage, Wage[-train,])

r2_score(Wage$wage[-train], yhat_test)
rmse(Wage$wage[-train], yhat_test)


# 3. Bagging ####
bag.wage = randomForest(wage ~ .-logwage, Wage, subset=train,
                          mtry = dim(Wage)[2]-2, ntree=25,
                          importance=TRUE)
bag.wage
yhat.bag = predict(bag.wage, newdata=Wage[-train,])

# Verifica os valores preditos e observados
plot(Wage$wage[-train], yhat.bag)
abline(0,1, col='red')

rmse(cst$Sales[-train], yhat.bag)

# 4. Random Forest ####
rf.wage = randomForest(wage~.-logwage, data=Wage,
                         subset=train, mtry=6, importance=TRUE)
yhat.rf = predict(rf.wage, newdata=Wage[-train,])
rmse(Wage$wage[-train], yhat.rf)

# Verificando Feature importances
importance(rf.wage)
varImpPlot(rf.wage)
