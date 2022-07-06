install.packages("devtools")
library('devtools')
install_github('dkahle/ggmap')
library('ggmap')
install.packages("magrittr")
install.packages('forecast', dependencies = TRUE)

library(ggplot2)
library(dplyr)
library(ggpubr)
library(forecast)
library(corrplot)
library(magrittr)
library(ggthemes)
library(gridExtra)



logist_data <- read.csv(file.choose(), header = TRUE)
head(logist_data)
names(logist_data) <- logist_data[1, ]
logist_data <- logist_data[-1, ]
rownames(logist_data) = NULL
head(logist_data)

logist_data <- logist_data[, -1]
summary(logist_data)


logist_data[, names(logist_data)] <- lapply(logist_data[, names(logist_data)], as.numeric) 

summary(logist_data)
logist_data %>% na.omit%>%cor %>% corrplot.mixed(p.mat=p.value[[1]], sig.level=.05, lower = 'number', upper='pie', tl.cex=.6, tl.col='black', order='hclust')

corrplot(cor(logist_data))

corrplot(cor(logist_data), 
         method="circle", order = "hclust", tl.pos = "n",
         shade.col=NA, tl.col="black", tl.srt=45)

cor(logist_data)
table(logist_data$SEX)
table(logist_data$EDUCATION)
table(logist_data$MARRIAGE) 
table(logist_data$`default payment next month`)
#logist_data -> 0, 5, 6 Á¦¿Ü => unknown data
logist_data <- subset(logist_data, (EDUCATION >= 1 & EDUCATION <= 4) &
                        (MARRIAGE >= 1 & MARRIAGE <= 2))

table(logist_data$SEX)
table(logist_data$EDUCATION)
table(logist_data$MARRIAGE) 
colnames(logist_data)[colnames(logist_data) == "PAY_0"] = "PAY_1"
colnames(logist_data)[colnames(logist_data) == "default payment next month"] = "default_payment"

summary(logist_data)
nrow(logist_data)
logist_data
logist_data$SEX <- as.factor(logist_data$SEX)
levels(logist_data$SEX) <- c("Male", "Female")

logist_data$EDUCATION <- as.factor(logist_data$EDUCATION)
levels(logist_data$EDUCATION) <- c("Graduate School", "University",
                                   "High School", "others")
logist_data$EDUCATION
logist_data$MARRIAGE <- as.factor(logist_data$MARRIAGE)
levels(logist_data$MARRIAGE) <- c("Married", "Single")

levels(logist_data$SEX)
levels(logist_data$EDUCATION)
levels(logist_data$MARRIAGE)
ggplot(data = logist_data, aes(x = SEX)) +
  geom_histogram(stat = "count", 
                 fill = "blue", colour = "black") +
  labs(title = 'SEX distribution',
       x = 'SEX',
       y = 'count',
       caption = 'SEX') +
  theme_solarized() +
  scale_y_continuous(n.breaks = 6)

ggplot(data = logist_data, aes(x = MARRIAGE)) +
  geom_histogram(stat = "count",
                 fill = "blue", colour = "black") +
  labs(title = 'MARRIAGE distribution',
       x = 'MARRIAGE',
       y = 'count',
       caption = 'MARRIAGE') +
  theme_solarized() +
  scale_y_continuous(n.breaks = 6)

ggplot(data = logist_data, aes(x = EDUCATION)) +
  geom_histogram(stat = "count",
                 fill = "blue", colour = "black") +
  labs(title = 'EDUCATION distribution',
       x = 'EDUCATION',
       y = 'count',
       caption = 'EDUCATION') +
  theme_solarized() +
  scale_y_continuous(n.breaks = 6)

ggplot(data = logist_data, aes(x = AGE)) +
  geom_histogram(stat = "count", fill = "blue") +
  scale_x_discrete(breaks = seq(20, 80, 5)) +
  xlim(20, 80) +
  scale_y_continuous(n.breaks = 5) +
  labs(title = 'age distribution',
       x = 'age',
       y = 'count',
       caption = 'age') +
  theme_solarized() 

ggplot(data = logist_data, aes(x = SEX, fill = as.factor(default_payment))) +
  geom_histogram(stat = "count", 
                 colour = "black") +
  labs(title = 'SEX - default payment distribution',
       x = 'SEX',
       y = 'count') +
  theme_solarized() +
  theme(title = element_text(size = 10)) +
  scale_y_continuous(n.breaks = 6)

ggplot(data = logist_data, aes(x = MARRIAGE, fill = as.factor(default_payment))) +
  geom_histogram(stat = "count",
                 colour = "black") +
  labs(title = 'MARRIAGE - default payment distribution',
       x = 'MARRIAGE',
       y = 'count') +
  theme_solarized() +
  theme(title = element_text(size = 10)) +
  scale_y_continuous(n.breaks = 6)

ggplot(data = logist_data, aes(x = EDUCATION, fill = as.factor(default_payment))) +
  geom_histogram(stat = "count",
                 colour = "black") +
  labs(title = 'EDUCATION - default payment distribution',
       x = 'EDUCATION',
       y = 'count') +
  theme_solarized() +
  theme(title = element_text(size = 10)) +
  scale_y_continuous(n.breaks = 6)

grid.arrange(a, b, c, nrow = 3)

ggplot(data = logist_data, aes(x = AGE, fill = as.factor(default_payment))) +
  geom_histogram(stat = "count") +
  scale_x_discrete(breaks = seq(20, 80, 5)) +
  xlim(20, 80) +
  scale_y_continuous(n.breaks = 5) +
  labs(title = 'age - default payment distribution',
       x = 'age',
       y = 'count',
       caption = 'age') +
  theme_solarized() 



grid.arrange(grobs = list(a, b, c, d), nrow = 4)
ggplot(data = logist_data) +
  geom_density(mapping = aes(x = LIMIT_BAL), color = "red",
               fill = "red") +
  labs(title = "LIMIT_BAL Density", x = "LIMIT_BAL", y = "Density")+
  theme_solarized() +
  scale_x_continuous(n.breaks = 5)

ggplot(data = logist_data) +
  geom_density(mapping = aes(x = LIMIT_BAL, fill = as.factor(default_payment))) +
labs(title = "LIMIT_BAL - default_payment",
       x = "LIMIT_BAL", y = "density") +
  scale_x_continuous(n.breaks = 5)+
  theme_solarized()

ggplot(data = logist_data, aes(y = LIMIT_BAL)) +
  geom_boxplot()

delay <- c('PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6')
bill_state <- c('BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4',
                'BILL_AMT5', 'BILL_AMT6')
pay_amount <- c('PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3',
                'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6')
summary(logist_data[, delay])
summary(logist_data[, bill_state])
summary(logist_data[, pay_amount])


table(logist_data$PAY_1)
tail(logist_data)
library(gridExtra)
plot_list <- list()
ggplot(logist_data, aes(x = PAY_2, fill = as.factor(default_payment))) + 
  geom_histogram(binwidth = 1) + 
  # I include education since we know (a priori) it's a significant predictor
  # facet_grid(.~EDUCATION) + 
  theme_fivethirtyeight()


ggplot(data = logist_data, aes(x = EDUCATION, fill = as.factor(default_payment))) +
  geom_histogram(stat = "count",
                 colour = "black") +
  labs(title = 'EDUCATION - default payment distribution',
       x = 'EDUCATION',
       y = 'count',
       caption = 'EDUCATION') +
  theme_solarized() +
  scale_y_continuous(n.breaks = 6)

ggplot(data = logist_data, aes(x = PAY_1, fill = as.factor(default_payment))) +
  geom_histogram(stat = "count",
                 colour = "black") +
  labs(title = 'PAY_1 - default payment distribution',
       x = 'PAY_1',
       y = 'count',
       caption = 'PAY_1') +
  theme_solarized() +
  scale_y_continuous(n.breaks = 6)

ggplot(data = logist_data, aes(x = PAY_2, fill = as.factor(default_payment))) +
  geom_histogram(stat = "count",
                 colour = "black") +
  labs(title = 'PAY_2 - default payment distribution',
       x = 'PAY_2',
       y = 'count',
       caption = 'PAY_2')+
  scale_x_discrete(limits = c(-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8)) +
  theme_solarized() +
  scale_y_continuous(n.breaks = 6)

ggplot(data = logist_data, aes(x = PAY_3, fill = as.factor(default_payment))) +
  geom_histogram(stat = "count",
                 colour = "black") +
  labs(title = 'PAY_3 - default payment distribution',
       x = 'PAY_3',
       y = 'count',
       caption = 'PAY_3') +
  scale_x_discrete(limits = c(-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8)) +
  theme_solarized() +
  scale_y_continuous(n.breaks = 6)

grid.arrange(a, b, c, nrow = 3)
ggplot(data = logist_data, aes(x = PAY_4, fill = as.factor(default_payment))) +
  geom_histogram(stat = "count",
                 colour = "black") +
  labs(title = 'PAY_4 - default payment distribution',
       x = 'PAY_4',
       y = 'count',
       caption = 'PAY_4') +
  scale_x_discrete(limits = c(-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8)) +
  theme_solarized() +
  scale_y_continuous(n.breaks = 6)

ggplot(data = logist_data, aes(x = PAY_5, fill = as.factor(default_payment))) +
  geom_histogram(stat = "count",
                 colour = "black") +
  labs(title = 'PAY_5 - default payment distribution',
       x = 'PAY_5',
       y = 'count',
       caption = 'PAY_5') +
  scale_x_discrete(limits = c(-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8)) +
  theme_solarized() +
  scale_y_continuous(n.breaks = 6)

ggplot(data = logist_data, aes(x = LIMIT_BAL, fill = as.factor(default_payment))) +
  geom_density(
                 colour = "black") +
  labs(title = 'LIMIT_BAL - default payment distribution',
       x = 'LIMIT_BAL',
       y = 'count',
       caption = 'LIMIT_BAL') +
  theme_solarized() +
  scale_y_continuous(n.breaks = 6)

ggplot(data = logist_data, aes(x = SEX, y = LIMIT_BAL,
                               fill = as.factor(SEX))) +
  geom_boxplot(
                 colour = "black") +
  labs(title = 'SEX - LIMIT_BAL box plot',
       x = 'SEX',
       y = 'LIMIT_BAL',
       caption = 'SEX - LIMIT_BAL') +
  theme_solarized() +
  scale_y_continuous(n.breaks = 6)
ggplot(data = logist_data, aes(x = MARRIAGE, y = LIMIT_BAL,
                               fill = as.factor(MARRIAGE))) +
  geom_boxplot(
    colour = "black") +
  labs(title = 'MARRIAGE - LIMIT_BAL box plot',
       x = 'MARRIAGE',
       y = 'LIMIT_BAL',
       caption = 'MARRIAGE - LIMIT_BAL') +
  theme_solarized() +
  scale_y_continuous(n.breaks = 6)

ggplot(data = logist_data, aes(x = EDUCATION, y = LIMIT_BAL,
                               fill = as.factor(EDUCATION))) +
  geom_boxplot(
    colour = "black") +
  labs(title = 'EDUCATION - LIMIT_BAL box plot',
       x = 'EDUCATION',
       y = 'LIMIT_BAL',
       caption = 'EDUCATION - LIMIT_BAL') +
  theme_solarized() +
  scale_y_continuous(n.breaks = 6)

  ggplot(data = logist_data, aes(x = BILL_AMT2, fill = as.factor(default_payment))) +
    geom_density(
      colour = "black") +
    labs(title = 'BILL_AMT2 - default payment distribution',
         x = 'BILL_AMT2',
         y = 'count',
         caption = 'BILL_AMT2') +
    scale_x_continuous(limits = c(0, 100000))+
  theme_solarized() +
    scale_y_continuous(n.breaks = 6)
  
  ggplot(data = logist_data, aes(x = BILL_AMT3, fill = as.factor(default_payment))) +
    geom_density(
      colour = "black") +
    labs(title = 'BILL_AMT3 - default payment distribution',
         x = 'BILL_AMT3',
         y = 'count',
         caption = 'BILL_AMT3') +
    scale_x_continuous(limits = c(0, 100000)) +
  theme_solarized() +
    scale_y_continuous(n.breaks = 6)
  
  ggplot(data = logist_data, aes(x = BILL_AMT4, fill = as.factor(default_payment))) +
    geom_density(
      colour = "black") +
    labs(title = 'BILL_AMT4 - default payment distribution',
         x = 'BILL_AMT4',
         y = 'count',
         caption = 'BILL_AMT4') +
    scale_x_continuous(limits = c(0, 100000)) +
  theme_solarized() +
    scale_y_continuous(n.breaks = 6)
  
  ggplot(data = logist_data, aes(x = BILL_AMT5, fill = as.factor(default_payment))) +
    geom_density(
      colour = "black") +
    labs(title = 'BILL_AMT5 - default payment distribution',
         x = 'BILL_AMT5',
         y = 'count',
         caption = 'BILL_AMT5') +
    scale_x_continuous(limits = c(0, 100000)) +
  theme_solarized() +
    scale_y_continuous(n.breaks = 6)
  
  ggplot(data = logist_data, aes(x = BILL_AMT6, fill = as.factor(default_payment))) +
    geom_density(
      colour = "black") +
    labs(title = 'BILL_AMT6 - default payment distribution',
         x = 'BILL_AMT6',
         y = 'count',
         caption = 'BILL_AMT6') +
    scale_x_continuous(limits = c(0, 100000)) +
  theme_solarized() +
    scale_y_continuous(n.breaks = 6)
  
  
  
  ggplot(data = logist_data, aes(x = PAY_AMT1, fill = as.factor(default_payment))) +
    geom_density(
      colour = "black") +
    labs(title = 'PAY_AMT1 - default payment distribution',
         x = 'PAY_AMT1',
         y = 'count',
         caption = 'PAY_AMT1') +
    scale_x_continuous(limits = c(0, 100000)) +
  theme_solarized() +
    scale_y_continuous(n.breaks = 6)
  
  ggplot(data = logist_data, aes(x = PAY_AMT2, fill = as.factor(default_payment))) +
    geom_density(
      colour = "black") +
    labs(title = 'PAY_AMT2 - default payment distribution',
         x = 'PAY_AMT2',
         y = 'count',
         caption = 'PAY_AMT2') +
    scale_x_continuous(limits = c(0, 100000))
  theme_solarized() +
    scale_y_continuous(n.breaks = 6)
  
  ggplot(data = logist_data, aes(x = PAY_AMT3, fill = as.factor(default_payment))) +
    geom_density(
      colour = "black") +
    labs(title = 'PAY_AMT3 - default payment distribution',
         x = 'PAY_AMT3',
         y = 'count',
         caption = 'PAY_AMT3') +
    scale_x_continuous(limits = c(0, 100000))
  theme_solarized() +
    scale_y_continuous(n.breaks = 6)
  
  ggplot(data = logist_data, aes(x = PAY_AMT4, fill = as.factor(default_payment))) +
    geom_density(
      colour = "black") +
    labs(title = 'PAY_AMT4 - default payment distribution',
         x = 'PAY_AMT4',
         y = 'count',
         caption = 'PAY_AMT4') +
    scale_x_continuous(limits = c(0, 100000))
  theme_solarized() +
    scale_y_continuous(n.breaks = 6)
  
  ggplot(data = logist_data, aes(x = PAY_AMT5, fill = as.factor(default_payment))) +
    geom_density(
      colour = "black") +
    labs(title = 'PAY_AMT5 - default payment distribution',
         x = 'PAY_AMT5',
         y = 'count',
         caption = 'PAY_AMT5') +
    scale_x_continuous(limits = c(0, 100000))
  theme_solarized() +
    scale_y_continuous(n.breaks = 6)
  
  ggplot(data = logist_data, aes(x = PAY_AMT6, fill = as.factor(default_payment))) +
    geom_density(
      colour = "black") +
    labs(title = 'PAY_AMT6 - default payment distribution',
         x = 'PAY_AMT6',
         y = 'count',
         caption = 'PAY_AMT6') +
    scale_x_continuous(limits = c(0, 100000))
  theme_solarized() +
    scale_y_continuous(n.breaks = 6)
  
table(logist_data$PAY_6)
grid.arrange(grobs = plot_list, nrow = 3, ncol = 2)
warnings()
summary(logist_data)
cor(logist_data)

cor.test(logist_data$SEX, logist_data$default_payment)
data%>%cor.mtest(method='pearson')->p.value
str(p.value)

table(logist_data$PAY_1)
table(logist_data$PAY_2)
table(logist_data$PAY_3)
table(logist_data$PAY_4)
table(logist_data$PAY_5)
table(logist_data$PAY_6)

delay <- c('PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6')

  logist_data[delay] <- lapply(logist_data[delay],
                             function(x) as.numeric(x))
table(logist_data$PAY_1)
payment <- logist_data[delay]
payment
logist_data$PAY_1 <- cut(payment$PAY_1, 
                         breaks = c(-2, -1, 0, 2, 3, 9),
                         include.lowest = TRUE,
                         right = FALSE,
                         labels = c(1, 2, 3, 4, 5))
logist_data$PAY_2 <- cut(payment$PAY_2, 
                         breaks = c(-2, -1, 0, 2, 3, 9),
                         include.lowest = TRUE,
                         right = FALSE,
                         labels = c(1, 2, 3, 4, 5))
logist_data$PAY_3 <- cut(payment$PAY_3, 
                         breaks = c(-2, -1, 0, 2, 3, 9),
                         include.lowest = TRUE,
                         right = FALSE,
                         labels = c(1, 2, 3, 4, 5))
logist_data$PAY_4 <- cut(payment$PAY_4, 
                         breaks = c(-2, -1, 0, 2, 3, 9),
                         include.lowest = TRUE,
                         right = FALSE,
                         labels = c(1, 2, 3, 4, 5))
logist_data$PAY_5 <- cut(payment$PAY_5, 
                         breaks = c(-2, -1, 0, 2, 3, 9),
                         include.lowest = TRUE,
                         right = FALSE,
                         labels = c(1, 2, 3, 4, 5))
logist_data$PAY_6 <- cut(payment$PAY_6, 
                         breaks = c(-2, -1, 0, 2, 3, 9),
                         include.lowest = TRUE,
                         right = FALSE,
                         labels = c(1, 2, 3, 4, 5))
logist_data

standard_scaler = function(x){
  result = (x - mean(x)) / sd(x)
  return(result)
}

names(logist_data)
logist_data
scale_set <- colnames(logist_data)[c(1, 5, seq(12, 23, 1))]
scale_set

logist_data[scale_set] <- lapply(logist_data[scale_set],
                             function(x) standard_scaler(x))

logist_data
apply(logist_data[scale_set], 2, mean)
apply(logist_data[scale_set], 2, sd)
table(logist_data$PAY_1)

set.seed(2022)
test_id <- sample(1:nrow(logist_data), nrow(logist_data) * 0.8)
dat_train <- logist_data[test_id, ]
dat_test <- logist_data[-test_id, ]
nrow(dat_train)
nrow(dat_test)

model_rg <- glm(default_payment~., dat_train, family = binomial())
summary(model_rg)

perf_eval <- function(cm){
  # true positive rate
  TPR = Recall = cm[2,2]/sum(cm[2,])
  # precision
  Precision = cm[2,2]/sum(cm[,2])
  # true negative rate
  TNR = cm[1,1]/sum(cm[1,])
  # accuracy
  ACC = sum(diag(cm)) / sum(cm)
  # balance corrected accuracy (geometric mean)
  BCR = sqrt(TPR*TNR)
  # f1 measure
  F1 = 2 * Recall * Precision / (Recall + Precision)
  
  re <- data.frame(TPR = TPR,
                   Precision = Precision,
                   TNR = TNR,
                   ACC = ACC,
                   BCR = BCR,
                   F1 = F1)
  return(re)
}

pred_probility <- c(0.18, 0.23, 0.28, 0.35, 0.4)
for (i in 1:5){
pred_prob <- predict(model_rg, dat_test, type ="response")
pred_class <- rep(0, nrow(dat_test))
pred_class[pred_prob > pred_probility[i]] <- 1
cm <- table(pred = pred_class, actual = dat_test$default_payment)
print(perf_eval(cm))
}

pred_prob <- predict(model_rg, dat_test, type ="response")
pred_class <- rep(0, nrow(dat_test))
pred_class[pred_prob > 0.28] <- 1
cm <- table(pred = pred_class, actual = dat_test$default_payment)
print(cm)
perf_eval(cm)


model_fwd <- step(glm(default_payment ~ 1, dat_train, 
                      family = binomial()), 
                  direction = "forward", trace = 0,
                  scope = formula(model_rg))

pred_prob <- predict(model_fwd, dat_test, type="response")
pred_class <- rep(0, nrow(dat_test))
pred_class[pred_prob > 0.28] <- 1
cm <- table(pred=pred_class, actual=dat_test$default_payment)
perf_eval(cm)

model_bwd <- step(glm(default_payment ~ ., dat_train, 
                      family = binomial()), 
                  direction = "backward", trace = 0,
                  scope = list(lower=default_payment ~ 1))
summary(model_bwd)
pred_prob <- predict(model_bwd, dat_test, type="response")
pred_class <- rep(0, nrow(dat_test))
pred_class[pred_prob > 0.28] <- 1
cm <- table(pred=pred_class, actual=dat_test$default_payment)
perf_eval(cm)

model_step <- step(glm(default_payment ~ ., dat_train,
                       family = binomial()), direction = "both", trace = 0,
                   scope = list(lower=default_payment ~ 1, upper = formula(model_rg)))

pred_prob <- predict(model_step, dat_test, type="response")
pred_class <- rep(0, nrow(dat_test))
pred_class[pred_prob > 0.28] <- 1
cm <- table(pred=pred_class, actual=dat_test$default_payment)
perf_eval(cm)

summary(model_rg)
names(dat_train)
model_rg <- glm(default_payment ~ LIMIT_BAL + SEX+ MARRIAGE +
                  PAY_1 + PAY_2 + PAY_3 + PAY_4 + PAY_5 + PAY_6 +
                 BILL_AMT3 + BILL_AMT4 +
                  BILL_AMT5 + BILL_AMT6 + PAY_AMT1 + PAY_AMT2 +
                  PAY_AMT3 + PAY_AMT4 + PAY_AMT5 + PAY_AMT6,
                dat_train, family = binomial())
summary(model_rg)

model_rg <- glm(default_payment ~ LIMIT_BAL + SEX+ MARRIAGE +
                  PAY_1 + PAY_2 + PAY_3 + PAY_4 + PAY_5 + PAY_6 +
                  BILL_AMT3 + PAY_AMT1 + PAY_AMT2 +
                  PAY_AMT3 + PAY_AMT6,
                dat_train, family = binomial())
summary(model_rg)

model_rg <- glm(default_payment ~ LIMIT_BAL + SEX+ MARRIAGE +
                  PAY_1 + PAY_2 + PAY_3 + PAY_4 + PAY_5 + PAY_6 +
                  BILL_AMT3 + BILL_AMT4 +
                   BILL_AMT6 + PAY_AMT1 + PAY_AMT2 +
                  PAY_AMT3 + PAY_AMT4 + PAY_AMT5 + PAY_AMT6,
                dat_train, family = binomial())
summary(model_rg)

model_rg <- glm(default_payment ~ LIMIT_BAL + SEX+ MARRIAGE +
                  PAY_1 + PAY_2 + PAY_3 + PAY_4 + PAY_5 + PAY_6 +
                  BILL_AMT3 + BILL_AMT4 +
                  BILL_AMT6 + PAY_AMT1 + PAY_AMT2 +
                  PAY_AMT3  + PAY_AMT5 + PAY_AMT6,
                dat_train, family = binomial())
summary(model_rg)

model_rg <- glm(default_payment ~ LIMIT_BAL + SEX+ MARRIAGE +
                  PAY_1 + PAY_2 + PAY_3 + PAY_4 + PAY_5 + PAY_6 +
                  BILL_AMT3 +
                  BILL_AMT6 + PAY_AMT1 + PAY_AMT2 +
                  PAY_AMT3  + PAY_AMT6,
                dat_train, family = binomial())
summary(model_rg)

model_rg <- glm(default_payment ~ LIMIT_BAL + SEX+ MARRIAGE +
                  PAY_1 + PAY_2 + PAY_3 + PAY_4 + PAY_5 + PAY_6 +
                  BILL_AMT3 + 
                   PAY_AMT1 + PAY_AMT2 +
                  PAY_AMT3  + PAY_AMT5 + PAY_AMT6,
                dat_train, family = binomial())
summary(model_rg)
model_rg <- glm(default_payment ~ .,
                dat_train, family = binomial())
pred_prob <- predict(model_rg, dat_test, type="response")
pred_class <- rep(0, nrow(dat_test))
pred_class[pred_prob > 0.28] <- 1
cm <- table(pred=pred_class, actual=dat_test$default_payment)
perf_eval(cm)
summary(model_step)
model_rg <- glm(default_payment ~ .,
                dat_train, family = binomial())
summary(model_step)
summary(model_rg)

library(pROC)
ROC <- roc(dat_test$default_payment, pred_class)
plot.roc(ROC, 
         col = "royalblue",
         print.auc = TRUE,
         max.auc.polygon = TRUE,
         print.thres = TRUE, print.thres.pch = 19, print.thres.col = "red",
         auc.polygon = TRUE, auc.polygon.col = "#A0A0A0")
model_rg <- glm(default_payment ~ .,
                dat_train, family = binomial())