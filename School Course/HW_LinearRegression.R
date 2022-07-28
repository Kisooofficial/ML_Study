install.packages("devtools")
library('devtools')
install_github('dkahle/ggmap')
library('ggmap')

install.packages("magrittr")
install.packages('forecast', dependencies = TRUE)
install.packages('ggthemes')

library(ggplot2)
library(dplyr)
library(ggpubr)
library(magrittr)
library(forecast)
library(corrplot)
library(ggthemes)
library(gridExtra)

data <- read.csv(file.choose(), header = 1)
head(data)
# date -> month, year로 나눴는데 없네? 그래서 month도 삭제

nrow(data)
ncol(data)

tail(data)

sum(is.na(data))

head(data)

data <- data[, -c(1, 17, 18, 19)] #id, zipcode 제거
head(data)


year <- as.numeric(format(as.Date(data$date, format = "%m/%d/%Y"), "%Y"))
month <- as.numeric(format(as.Date(data$date, format = "%m/%d/%Y"), "%m"))
year
month

data[, "year"] <- year
data[, "month"] <- month
head(data)
data[, "year_age"] <- 2022 - data$yr_built
data[, "year_renovation"] <- 2022 - data$yr_renovated
data[, "rooms"] <- data$bedrooms + data$bathrooms
head(data)

data <- data[, -c(1)]
head(data)

str(data)

library(corrplot)
library(magrittr)

cor(data)

data%>%cor.mtest(method='pearson')->p.value
str(p.value)

data %>% na.omit%>%cor %>% corrplot.mixed(p.mat=p.value[[1]], sig.level=.05, lower = 'number', upper='pie', tl.cex=.6, tl.col='black', order='hclust')

corrplot(cor(data))

cor(data)
corrplot(cor(data), 
         method="shade", 
         shade.col=NA, tl.col="black", tl.srt=45)
#price와 연관성 높은것들 -> bathrooms, sqft_living, grade, sqft_above,
#sqft_living15 + rooms + floors + waterfront + view + saft_basement
data[, "log_price"] <- log(data$price)
data

theme_set(theme_bw())
theme_set(theme_grey())

a <- ggplot(data = data) +
  geom_density(mapping = aes(x = price), color = "blue",
               fill = "red") +
  labs(title = "price density distribution",
       subtitle = "price density",
       caption = "price",
       x = "price (dollars)",
       y = "density") +
  theme_solarized()

data[, "log_price"] <- log(data$price)
b <- ggplot(data = data) +
  geom_density(mapping = aes(x = log_price), color = "red",
               fill = "blue") +
    labs(title = "log price density distribution",
         subtitle = "log price density",
         caption = "log price",
         x = "price (dollars)",
         y = "density") +
    theme_solarized()

grid.arrange(a, b, nrow = 2, ncol = 1)
hist(data$bedrooms, col = "blue",breaks = 12,
                      xlab= "방 개수")

data <- data[!(data$bedrooms >= 7), ]

max(data$bedrooms)
min(data$bedrooms)

ggplot(data, aes(x = bedrooms)) +
  geom_bar(stat = 'count', width = 1,
           fill = 'steelblue') +
  labs(title = "bedrooms density distribution",
       subtitle = "bedroom density",
       caption = "bedroom",
       x = "bedrooms", y = "density") +
  scale_x_discrete(limits = c(0, 1, 2, 3, 4, 5, 6)) +
  theme_stata()
table(data$view)
mean(data$bedrooms)

plotdata <- data %>%
  count(view) %>%
  arrange(desc(view)) %>%
  mutate(prop = round(n * 100 / sum(n), 1))


plotdata
ggplot(plotdata, aes(x = "", y = prop, fill = view)) +
  geom_bar(stat = "identity", width = 1, color = "black") +
  labs(title = "view pie chart distribution",
       subtitle = "view pie chart",
       caption = "view") +
  coord_polar("y", start = 0, direction = -1) +
  theme_solarized()

ggplot(data, aes(x = view)) +
  geom_bar(stat = 'count', fill = "red", width = 0.5, colour = "black") +
  labs(title = "view histogram distribution",
       subtitle = "view histogram",
       caption = "view",
       x = "view") +
  theme(legend.position = "FALSE") +
  theme_stata()
 
  
ggplot(data, aes(x = floors)) +
  geom_bar(stat = 'count', fill = "#F8766D", width = 0.5, 
           colour = "black") +
  labs(title = "floor histogram distribution",
       subtitle = "floor histogram",
       caption = "floor",
       x = "floor") +
  scale_x_continuous(breaks = seq(1, 3.5, 0.5)) +
  theme_solarized()

ggplot(data, aes(x = log_price)) +
  geom_histogram(fill = "blue", binwidth = 0.5, colour= "black") +
  labs(title = "floor - log_price histogram distribution",
       subtitle = "floor - log_price histogram",
       caption = "floor - log_price",
       x = "log_price") +
  ggtitle("comparison of floor-price") +
  facet_grid(. ~as.factor(floors)) +
  theme_solarized()

data %>% group_by(floors) %>% summarise(mean= mean(log_price))

exp(0.7)
str(data$floors)
ggplot(data = data) +
  geom_boxplot(aes(y = log_price), color = "blue") +
  theme_bw() +  
  labs(y = "log_price") + ggtitle("log화 한 가격")

ggplot(data, aes(x = log_price)) + 
  geom_bar(fill = "blue", colour = "red") +
  labs(title = "floor - log_price histogram distribution",
       subtitle = "floor - log_price histogram",
       caption = "floor - log_price",
       x = "log_price") +
  ggtitle("comparison of floor-price") +
  facet_grid(. ~as.factor(year)) +
  theme_solarized()

ggplot(data, aes(x = log_price)) + 
  geom_bar(fill = "blue", colour = "red") +
  labs(title = "year - log_price histogram distribution",
       subtitle = "year - log_price histogram",
       caption = "year - log_price",
       x = "log_price") +
  ggtitle("comparison of year-price") +
  facet_grid(. ~as.factor(year)) +
  theme_solarized()
data %>% group_by(year) %>% summarise(mean= mean(log_price))

year_month <- data %>% group_by(year, month) %>% summarise(mean = mean(log_price))
year_month$time <- paste(as.character(year_month$year),
                         as.character(year_month$month))
ggplot(year_month, aes(x = time, y = mean)) +
  geom_bar(stat = "identity", fill = "blue", colour = "black") +
  labs(title = "time - log_price barplot",
       caption = "time - log_price",
       x = "log_price") +
  ggtitle("comparision of time - price") +
  theme_solarized()
exp(0.1)
ggplot(data, aes(x = month)) + 
  geom_histogram(fill = "blue", colour = "black", bins = 12) +
  labs(title = "year house purchase", x = "month", y = "count") +
  scale_x_discrete(limits = c(1:12)) + ylim(c(0, 2500))

data
data <- subset(data, select = -c(year, month))
ggplot(data, aes(x = condition)) +
  geom_histogram(fill = "#F8766D", binwidth = 1, colour = "black") +
  ggtitle("condition histogram")

ggplot(data, aes(x = grade)) +
  geom_histogram(fill = "#F8766D", binwidth = 1, colour = "black") +
  ggtitle("grade histogram")

ggplot(data = data, aes(x = as.factor(floors), y= log_price,
                        group = as.factor(floors), fill = as.factor(floors))) +
  geom_boxplot() + scale_fill_brewer(palette = "Blues")

table(data$waterfront)

ggplot(data = data, aes(x = as.factor(month), y = log_price,
                        group = as.factor(month), fill = as.factor(month))) +
  geom_boxplot() + scale_fill_brewer(palette = "Blues")

data$month <- factor(data$month)
data %>% group_by(month) %>% summarise(mean = mean(price)) %>%
  ggbarplot(x = "month", y = "mean", fill = "month",
            palette = "Set3", size = 1, width = 1) +
  scale_y_continuous() +
  theme(legend.position = "none") + 
  labs(title = "Price Average By Month")

data %>% group_by(floors) %>% summarise(mean = mean(price)) %>%
  ggbarplot(x = "floors", y = "mean", fill = "floors",
            palette = "Set3", size = 2, bin_width = 0.5) +
  labs(title = "Price Average By floor",
       caption = "floor - price",
       x = "floors",
       y = "price") +
  scale_y_continuous(n.breaks = 10) +
  theme(legend.position = "none") + 
  theme_solarized()
cor(data)

data %>% group_by(floors) %>% summarise(mean = mean(price)) %>%
  ggbarplot(x = "floors", y = "mean", fill = "floors",
            palette = "Set3", size = 2, bin_width = 0.5) +
  scale_y_continuous(n.breaks = 10) +
  theme(legend.position = "none") + 
  labs(title = "Price Average By floor")

p <- data %>% group_by(bathrooms) %>% summarise(mean = mean(log_price)) %>%
  ggbarplot(x = "bathrooms", y = "mean" ,fill = "bathrooms",
             size = 1, width = 1) +
  labs(title = "bathrooms Average log price",
       caption = "bathrooms - log price",
       x = "bathrooms",
       y = "log_price") +
  theme_solarized()
p + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1))  

#data %>% group_by(bathrooms) %>% summarise(mean = mean(price)) %>%
  ggplot(data = data, aes(x = bathrooms, y = log_price, color = bathrooms),
         main = "Price Average By Bathrooms") +
  geom_point() +
  scale_y_continuous(n.breaks = 10) +
  labs(title = "bathrooms Average log price",
       caption = "bathrooms - log price",
       x = "bathrooms",
       y = "log_price") +
  theme_solarized()
  stat_smooth(color = 'red', fill = 'green')

data$grade <- as.factor(data$grade)
data %>% group_by(grade) %>% summarise(mean = mean(log_price)) %>%
  ggbarplot(x = "grade", y = "mean", fill = "grade",
            palette = "Set3", size = 2, bin_width = 0.5) +
  scale_y_continuous(labels = scales::dollar, n.breaks = 7) +
  labs(title = "grade Average log price",
       caption = "grade - log price",
       x = "grade",
       y = "log_price") +
  theme_solarized()

ggscatter(data,"sqft_living","price",color = "green",size=0.5,alpha=0.5) +
  geom_smooth(method="lm", col="darkred",size=2)+
  xlim(0,5000)+labs(x = 'living space', y = 'Price',title = "Price By living Space")+
  scale_y_continuous(n.breaks = 10)+coord_cartesian(y=c(0,3000000)) +
  labs(title = "living space & price scatter plot",
       caption = "living space - price",
       x = "living space",
       y = "price") +
  theme_solarized()

 ggplot(data, aes(x=sqft_living, y = log_price, color = bathrooms),
       main = "log_price by living space") +
  geom_point() +
  stat_smooth(color = "red", fill = "green") +
  labs(title = "sqft_living - log_price - bathrooms plot") +
  theme_solarized_2() 

 ggplot(data, aes(x=sqft_living, y = log_price, color = grade)) +
   geom_point() +
   stat_smooth(color = "red", fill = "green") +
   labs(title = "sqft_living - log_price - grade plot") +
   theme_solarized_2() 

ggplot(data, aes(x=sqft_living15, y = log_price, color = condition)) +
  geom_point() +
  stat_smooth(color = "red", fill = "green") +
  labs(title = "sqft_living15 - log_price - condition plot") +
  theme_solarized_2() 

ggplot(data, aes(x=bathrooms, y = price, color = "blue"),
       main = "Price by living space") +
  geom_point() +
  stat_smooth(color = "red", fill = "green")

data$grade <- as.factor(data$grade)
data %>% group_by(grade) %>% summarise(mean = mean(price)) %>%
  ggbarplot(x = "grade", y = "mean", fill = "grade",
            palette = "Set3", size = 2, bin_width = 0.5) +
  scale_y_continuous(labels = scales::dollar, n.breaks = 15) +
  theme(legend.position = "none") + 
  labs(title = "Price Average By grade")

data$grade <- as.numeric(data$grade)
boxplot(log_price~floors, data = data,
        col = rainbow(7),
        xlab = "층 수",
        ylab = "가격",
        main = "층 수별 가격대")
#price와 연관성 높은것들 -> bathrooms, sqft_living, grade, sqft_above,
#sqft_living15 + rooms + floors + waterfront + view + saft_basement

apply(data, 2, mean) 
apply(data, 2, sd)

standard_scaler = function(x){
  result = (x - mean(x)) / sd(x)
  return(result)
}

names(data)
scale_set <- colnames(data)[-c(1, 7, length(colnames(data)))]
scale_set

data$waterfront <- as.factor(data$waterfront)
data$waterfront
data_scale <- as.data.frame(apply(X = data[scale_set], MARGIN = 2, FUN = "standard_scaler"))
data_scale <- cbind(data_scale, data$waterfront, data$log_price)
names(data_scale)[length(colnames(data_scale))] <- c("log_price")
names(data_scale)[length(colnames(data_scale))] <- c("waterfront")
data_scale
data_scale <- data_scale[, -17]
data_price <- subset(data, select = -log_price)

waterfront_1 <- (data_scale$waterfront == 1) * 1
waterfront_1
data
lm1 <- lm(price ~., data = data_price)
summary(lm1)
data$waterfront <- as.numeric(data$waterfront)
cor(data)
lm2 <- lm(log_price ~., data = data_scale)
summary(lm2)
qt(0.01, 21520)
lm2 <- lm(log_price ~., data = data_scale)
summary(lm2)
sd(data_scale$sqft_basement)
par(mfrow = c(2, 1))
plot(data$price)
lines(lm1$fitted.values, col = "red")

plot(data_scale$log_price)
lines(lm2$fitted.values, col = "red")

summary(lm1)
accuracy(lm1)
accuracy(lm2)

install.packages("Metrics")
library(Metrics)
pred = predict(lm2, data = data_scale)
mse(data$log_price, pred)
mae(data$log_price, pred)
#변수 선택법
model1 <- lm(log_price~., data = data_scale)
summary(model1)
#floors > view > bedrooms > bathrooms > sqft_above > sqft_living15
#> sqft_living > grade
model1 <- lm(log_price~floors, data_scale)
summary(model1)

model2 <- lm(log_price~floors + view, data = data_scale)
summary(model2)

anova(model1, model2)
model3 <- lm(log_price ~ floors + view + bedrooms, data = data_scale)
summary(model3)

model4 <- lm(log_price~floors + view + bedrooms + bathrooms, 
             data = data_scale)
summary(model4)

model5 <- lm(log_price~floors + view + bedrooms +
               bathrooms + condition, data = data_scale)
summary(model5)
anova(model4, model5)

model6 <- lm(log_price~floors + view + bedrooms +
               bathrooms + sqft_above, data = data_scale)
summary(model6)
anova(model4, model6)
model7 <- lm(log_price~floors + view + bedrooms +
               bathrooms + sqft_above + sqft_living15,
             data = data_scale)
summary(model7)

model8 <- lm(log_price~floors + view + bedrooms +
               bathrooms + sqft_above + sqft_living15 + 
               sqft_living,
             data = data_scale)
summary(model8)

anova(model6, model9)
model9 <- lm(log_price~floors + view + bedrooms +
               bathrooms + sqft_above + sqft_living15 + 
               sqft_living + grade,
             data = data_scale)
summary(model9)

model1 <- lm(log_price~bedrooms + bathrooms + sqft_living +
               floors + waterfront + view + condition + grade +
               sqft_above + yr_built + sqft_living15 +sqft_lot15, data = data_scale)
summary(model1)

model1 <- lm(log_price~bedrooms + bathrooms + sqft_living +
               floors + waterfront + view + condition + grade +
               sqft_above + yr_built + sqft_living15, data = data_scale)
summary(model1)
model1 <- lm(log_price ~., data = data_scale)
summary(model1)
model2 <- lm(log_price ~ bathrooms + sqft_living +
               floors  + view + condition + grade +
               sqft_above + yr_built + sqft_living15 
             + waterfront, data = data_scale)
summary(model2)

data_scale
anova(model1, model2)
cor(data_scale)
dev.off()
linear_model <- lm(log_price ~ bathrooms, data = data)
summary(linear_model)

multi_model <- lm(log_price ~ bathrooms + grade, data = data)
summary(multi_model)

anova(linear_model, multi_model)

multi_model2 <- lm(log_price ~ bathrooms + grade + sqft_living,
                   data = data)
summary(multi_model2)

multi_model3 <- lm(log_price ~ bedrooms +
                     grade + sqft_living + condition, data = data)
summary(multi_model3)

anova(multi_model2, multi_model3)

multi_model4 <- lm(log_price ~ bathrooms + floors + sqft_living +
                     condition + grade + sqft_above, data = data)
summary(multi_model4)

data
