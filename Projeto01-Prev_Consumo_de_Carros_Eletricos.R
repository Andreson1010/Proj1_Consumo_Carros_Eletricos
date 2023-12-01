
#setwd("E:/DataScience/FCD/BigDataRAzure/Cap20_Projetos_com_FeedBack/Proj01_PrevConsumoCarrosEletricos")
#getwd()

# ====Definição do Problema======:

# Construir um modelo de Machine Learning capaz de prever o consumo
# de energia de veículos elétricos.


# =====Dicionário de Dados======:

# $ Car_full_name   - Nome do veículo         
# $ Make            - Fabricante      
# $ Model           - Modelo       
# $ Minimal_price   - Preço Mínimo na Polônia     
# $ Engine_power    - Potência        
# $ Maximum_torque  - Torque Máximo  (Nm)      
# $ Type_of_brakes  - Tipo de Freio: a disco ou tambor        
# $ Drive_type      - Tipo de Tração: trazeira, dianteira ou 4x4      
# $ Battery_capacity - Capacidade da Bateria KWh       
# $ Range            - Autonomia em Km       
# $ Wheelbase        - Distância entre os centros das rodas trazeira e dianteira       
# $ Length           - Comprimento do veículo em centímetros      
# $ Width            - Largura do veículo em centímetros   
# $ Height           - Altura do veículo  em centímetros      
# $ Minimal_empty_weight  - Peso próprio (kg)  
# $ Permissable_gross_weight - Peso total máximo (kg)
# $ Maximum_load_capacity - Capacidade de carga (kg)
# $ Number_of_seats -  Número de assentos         
# $ Number_of_doors -  Número de portas      
# $ Tire_size      - Dimensão do pneu em polegadas      
# $ Maximum_speed - Velocidade máxima em kph      
# $ Boot_capacity - Capacidade de Bagagem - Seats Up (litros, método VDA)           
# $ Acceleration  - Aceleração 0 -> 100 km/h (seg)      
# $ mumm_DC_charging_power - Carregamento DC (tipo de plugue / máx. kW) 
# $ mean_Energy_consumption - Consumo Médio de Energia - Ciclo Combinado (kWh / 100km)

library(dplyr)
library(readxl)
library(Amelia)
library(ggplot2)

#######Carga dos Dados##########:

df <- read_excel("EletricCar.xlsx")

# Dimensão dos dados
dim(df)

# Visualiza os Dados
View(df)

# Variáveis e Tipos dos dados
str(df)

# Estatisticas das Variáveis numéricas
summary(df)


##### Análise Exploratória dos Dados - Limpeza dos Dados ##### 

# Renomeando as colunas
colnames(df) <- c("Car_full_name",
                  "Make",
                  "Model",
                  "Minimal_price",
                  "Engine_power",
                  "Maximum_torque",
                  "Type_of_brakes",
                  "Drive_type",
                  "Battery_capacity",
                  "Range",
                  "Wheelbase",
                  "Length",
                  "Width",
                  "Height",
                  "Minimal_empty_weight",
                  "Permissable_gross_weight",
                  "Maximum_load_capacity",
                  "Number_of_seats",
                  "Number_of_doors",
                  "Tire_size",
                  "Maximum_speed",
                  "Boot_capacity",
                  "Acceleration",
                  "Maximum_DC_charging_power",
                  "mean_Energy_consumption")

# Verifica o resultado da renomeação das colunas
colnames(df)

# Quantas linhas tem casos completos?
complete_cases <- sum(complete.cases(df))
complete_cases

not_complete_cases <- sum(!complete.cases(df))
not_complete_cases

# Qual o percentual de dados incompletos?

percentual <- (not_complete_cases / complete_cases) * 100
percentual # R: 26%


sum(is.na(df)) # existem 30 valores NA


# A decisão foi retirar os valores NA. O dataset permneceu com  42 linhas
# e 25 colunas.

df_semNA <- na.omit(df)

str(df_semNA)


# Retirandoa as variáveis categóricas da coluna 1 a 3, 7 e 8, por inferir que não contribuem para o consumo
# de conbustível.

colunas_categoricas <- sapply(df_semNA,is.character)
colunas_categoricas

df_num <- df_semNA[,!colunas_categoricas]
View(df_num)
str(df_num)

# Filtrando as colunas numéricas para verificar a correlação das variáveis:

data_cor <- cor(df_num)
View(data_cor)
head(data_cor)

library(corrplot)

corrplot(data_cor, method = 'color')

corrgram(df_num, order = TRUE, lower.panel = panel.shade,
         upper.panel = panel.pie, text.panel = panel.txt)


# ============Construindo o modelo V1===================================================== 

# Divisão dos dados:

library(caret)

split <- createDataPartition(y = df_num$mean_Energy_consumption, p = 0.7, list = FALSE)

# Criando os Dados de treino e teste:

dados_treino <- df_num[split,]
dados_teste <- df_num[-split,]


# Todas as variáveis preditoras foram utilizadas com algoritimo de regressão
# linear(lm)

modelo_V1 <- train(mean_Energy_consumption ~., data = dados_treino, method = "lm")

summary(modelo_V1)

# # Obteve-se R2 =  0.9773

# verificando a importancia das variáveis preditoras com o modelo treinado.

varImp(modelo_V1)

plot(varImp(modelo_V1))

#===================Construindo o modelo V2 ==========================:

# Variáveis descartdas após analise de varimp:

# Maximum_DC_charging_power 2.726e+00
# Minimal_price             4.268e-03
# Acceleration              0.000e+00

df_var_desc <- df_num[,- c(1,18,19)]
str(df_var_desc)
View(df_var_desc)

library(caTools)

# Criando os dados de treino e de teste:

split_V2 <- sample.split(df_var_desc$mean_Energy_consumption, SplitRatio = 0.70)

dados_treino2 <- subset(df_var_desc, split_V2 == TRUE)
dados_teste2 <- subset(df_var_desc, split_V2 == FALSE)

#treinado o modelo V2:

modelo_V2 <- train(mean_Energy_consumption ~., data = dados_treino2, method = "lm")

summary(modelo_V2)

# Multiple R-squared:  0.9858


# =========Construindo o modelo V3 ==================================================

# Aplicando normalização dos dados para tentar melhorar a acuracia do modelo.Normalizando os dados:
maxs <- apply(df_var_desc, 2, max)
mins <- apply(df_var_desc, 2, min)

# normalizando:

df_norm <- as.data.frame(scale(df_var_desc, center = mins, scale = maxs - mins ))
head(df_norm)
class(df_num_norm)
View(df_num_norm)

# Divisão dos dados normalizados:

split_norm <- createDataPartition(y = df_norm$mean_Energy_consumption, p = 0.7, list = FALSE)

# Criando os Dados de treino e teste:

dados_treino3 <- df_norm[split_norm,]
dados_teste3 <- df_norm[-split_norm,]

# Treinando o modelo novamente, mudando desta vez o algorítimo para Random 
# forest(rf):

modelo_V3 <- train(mean_Energy_consumption ~ ., data = dados_treino3, method = "rf")

summary(modelo_V3)

residuo <- residuals(modelo_V3)
residuo

# SSE -  Sum squared error

SSE = sum((residuo)^2)
SSE

# SST- Sum squared total
SST = sum((mean(df_norm$mean_Energy_consumption) - df_norm$mean_Energy_consumption)^2)
SST
R2 = 1 - (SSE/SST)

R2 # = 0.982456 # Piorou muito pouco em relação aos outros modelos


#==============Construindo o modelo V4 ======================================

# susbstituindo os valores NA pela média dos valores:


hist(na.omit(df$Permissable_gross_weight))


dfMedia <- df %>% 
  mutate(
    Permissable_gross_weight = coalesce( 
      Permissable_gross_weight,
      mean(
        Permissable_gross_weight,
        na.rm = TRUE
      )
    ),
    Maximum_load_capacity = coalesce( 
      Maximum_load_capacity,
      mean(
        Maximum_load_capacity,
        na.rm = TRUE
      )
    ),
    Boot_capacity = coalesce(
      Boot_capacity, mean(
        Boot_capacity, na.rm = TRUE
      )
    ),
    Acceleration = coalesce(
      Acceleration, mean(
        Acceleration, na.rm = TRUE
      )
    ),mean_Energy_consumption = coalesce(
      mean_Energy_consumption, mean(
        mean_Energy_consumption, na.rm = TRUE)
    )
    
  )

# Excluindo o único valor Na da coluna " Type of Break " 

dfMedia <- na.omit(dfMedia)
str(dfMedia)

# Excluindo as variáveis categóricas:

colunas_categoricas <- sapply(dfMedia,is.character)
colunas_categoricas
dfMedia_num <- dfMedia[,!colunas_categoricas]

str(dfMieda_num)

View(dfMedia_num)

# Divisão dos dados em treino e teste:

library(caret)

split_V4 <- createDataPartition(y = dfMedia_num$mean_Energy_consumption, p = 0.7, list = FALSE)

# Criando os Dados de treino e teste:

dados_treino4 <- dfMedia_num[split_V4,]
dados_teste4 <- dfMedia_num[-split_V4,]


# Utilizando todas as variáveis preditoras com algoritimo de regressão
# linear(lm), para treinar o modelo:

modelo_V4 <- train(mean_Energy_consumption ~., data = dados_treino4, method = "lm")

summary(modelo_V4) # Multiple R-squared:  0.9352


varImp(modelo_V4)

plot(varImp(modelo_V4))


#====Contruindo o modelo V5========================================================


# Aplicando normalização dos dados para tentar melhorar a acuracia do modelo.

maxs <- apply(dfMedia_num, 2, max)
mins <- apply(dfMedia_num, 2, min)

# normalizando:

dfMedia_norm <- as.data.frame(scale(dfMedia_num, center = mins, scale = maxs - mins ))

View(dfMedia_norm)

# treinado o modelo V5 com dados normalizados:

modelo_V5 <- train(mean_Energy_consumption ~., data = dfMedia_norm, method = "lm")

summary(modelo_V5) # Multiple R-squared:  0.9307



# O modelo escolhido foi o V2, por apresntar  Multiple R-squared:  0.9858


## ===========Teste do modelo escolhido (V2)==========================

valorprev <- predict(modelo_V2, dados_teste2)
valorprev


# Calculando o Mean Squared Error

MSE.nn <- sum((dados_teste2$mean_Energy_consumption - valorprev)^2)/nrow(dados_teste2)
MSE.nn #3.595636

# Comparando valores previstos e observados:

resultados <- as.data.frame(cbind(valorprev, dados_teste2$mean_Energy_consumption))

colnames(resultados) <- c("Previsto", "Real")

View(resultados)

plot(valorprev, dados_teste2$mean_Energy_consumption)


# Plot dos erros

ggplot(resultados, aes(x = Real,y = Previsto)) + 
  geom_point() + stat_smooth()




