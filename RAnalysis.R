install.packages("metrica")
library(metrica)

set.seed(183)
# Two-class
binomial_case <- data.frame(labels = sample(c("True","False"), 100, replace = TRUE),
                            predictions = sample(c("True","False"), 100, replace = TRUE))
# Multi-class
multinomial_case <- data.frame(labels = sample(c("Red","Blue", "Green"), 100,
                                               replace = TRUE), predictions = sample(c("Red","Blue", "Green"), 100, replace = TRUE))

# Plot two-class confusion matrix
confusion_matrix(data = binomial_case, obs = labels, pred = predictions, 
                 plot = TRUE, colors = c(low="pink" , high="steelblue"), unit = "count")

# Plot multi-class confusion matrix
confusion_matrix(data = multinomial_case, obs = labels, pred = predictions, 
                 plot = TRUE, colors = c(low="#f9dbbd" , high="#735d78"), unit = "count")


library(RColorBrewer)
cls=c(1092, 2218, 1624, 2322,1288)
coul <- brewer.pal(5, "Set2") 
names=c("negative","somewhat negative","neutral","somewhat positive","positive.")

par(mar=c(4,4,1,1))
barplot(height=cls, names=names, col=coul ) 

threshold = c(0.7,0.72,0.73,0.74,0.75,0.76)
tacc = c(0.788,0.788,0.791,0.789,0.789,0.794)
models =c(4,7,7,11,12,13)

tfl="/adt_geocomplex/bert_allresult/testRes.csv"
dt=read_csv(tfl)
dt=dt[order(dt$threshold),] 

par(mar=c(4,4,1,4)+0.3)
plot(dt$threshold,dt$Leadboard,type="l",col="red",xlab="Threshold (total score)",
     ylab="Accuracy/Correlation/Total score") 
points(dt$threshold,dt$Leadboard,col="red") 
lines(dt$threshold,dt$SST) 
lines(dt$threshold,dt$PATA) 
lines(dt$threshold,dt$STS)  
legend(0.725,0.794,lty=c(1),col=c("red"),legend="Dev score/acc/cor",bty="n",
       pt.cex = 0.4,cex=0.8)

par(new = TRUE,xpd = TRUE)          
plot(dt$threshold,dt$model,  col = "blue",type="l", axes = FALSE, xlab = "", ylab = "") 
points(dt$threshold,dt$model,col="blue")
axis(side = 4, at = c(4,6,8,10,12))       
mtext(" ", side = 4, line = 3,srt = 270)
#ylab="Numder of qualified models"
legend(0.725,12.5,lty=c(1),col=c("blue"),legend="Number of models",bty="n",
       pt.cex = 0.4,cex=0.8)

###################
par(mar=c(4,4,1,4)+0.3)
plot(dt$threshold,dt$SST,type="l",col="red",xlab="Threshold (total score)",
     ylab="Accuracy/Correlation/Total score") 
points(dt$threshold,dt$Leadboard,col="red") 
lines(dt$threshold,dt$SST) 
lines(dt$threshold,dt$PATA) 
lines(dt$threshold,dt$STS)  

par(mar=c(4,4,1,1)+0.3)
plot(dt$threshold,dt$PATA,type="l",col="blue",xlab="Threshold (total score)",
     ylab="Accuracy") 
lines(dt$threshold,dt$STS,col="green")   
legend(0.70,0.902,lty=c(1,1),col=c("blue","green"),legend=c("Paraphrase","Textual similarity"),bty="n",
       pt.cex = 0.4,cex=0.8,y.intersp=1.8 )


par(mar=c(4,4,1,1)+0.3)
plot(dt$threshold,dt$SST,type="l",col="orange",xlab="Threshold (total score)",
     ylab="SST Dev accuracy") 








x <- rnorm(45) 
y1 <- x + rnorm(45) 
y2 <- x + rnorm(45, 7) 

# Draw first plot using axis y1 
par(mar = c(7, 3, 5, 4) + 0.3)               
plot(x, y1, pch = 13, col = 2)   

# set parameter new=True for a new axis 
par(new = TRUE)          

# Draw second plot using axis y2 
plot(x, y2, pch = 15, col = 3, axes = FALSE, xlab = "", ylab = "") 

axis(side = 4, at = pretty(range(y2)))       
mtext("y2", side = 4, line = 3)







