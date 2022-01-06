# awk '{print $4}' ./hinnymeta3.fa.tb.tsv > ./hinnymeta3.fa.tb.len.tsv
# sed -i 's/.\{4\}//' hinnymeta3.fa.tb.len.tsv

rm(list = ls())
setwd("D:\\CurrentProjects\\AI+yao\\metabolism\\network")
filename="uniprot_trembl_bacteria.enzyme.tsv_protein.len"
data=read.table(filename)

min_val=min(data[,1])
max_val=max(data[,1])
breaks=c(0,100,200,300,400,500,600,700,800,900,1000,1500,2000,2500,3000,4000,5000,6000,max_val)

a=hist(data[,1],col='lightblue',labels=TRUE,breaks=breaks,freq=T)

# ## draw pdf
# pdf(paste(filename,"pdf",sep="."),13)
# hist(data[,1],xlab="contig_length",ylab="hist_of_EASmeta6_contigs",col='lightblue',labels=TRUE,ylim=c(0,max(a$counts)*1.1),breaks=breaks,freq=T)
# ## 绘制正态曲线
# x=data[,1]
# xfit <- seq(min(x),max(x),length=40)
# meannum <- mean(x,na.rm=TRUE)
# sdnum <- sd(x,na.rm=TRUE)
# yfit <- dnorm(xfit,mean = meannum,sd=sdnum)
# yfit <- yfit*diff(a$mids[1:2]*length(x))
# lines(xfit,yfit,col="black",lwd=2)
# dev.off()

cat(paste(filename," summary status :\n"),file=paste(filename,".summary.status",sep=""))
cat("\nmin:\n",file=paste(filename,".summary.status",sep=""),append = TRUE)
cat(min_val,file=paste(filename,".summary.status",sep=""),append = TRUE)
cat("\nmax:\n",file=paste(filename,".summary.status",sep=""),append = TRUE)
cat(max_val,file=paste(filename,".summary.status",sep=""),append = TRUE)
cat("\nbreaks:\n",file=paste(filename,".summary.status",sep=""),append = TRUE)
cat(breaks,file=paste(filename,".summary.status",sep=""),append = TRUE)
cat("\ncounts:\n",file=paste(filename,".summary.status",sep=""),append = TRUE)
cat(a$counts,file=paste(filename,".summary.status",sep=""),append = TRUE)
cat("\ndensity:\n",file=paste(filename,".summary.status",sep=""),append = TRUE)
cat(a$density,file=paste(filename,".summary.status",sep=""),append = TRUE)
cat("\nmids:\n",file=paste(filename,".summary.status",sep=""),append = TRUE)
cat(a$mids,file=paste(filename,".summary.status",sep=""),append = TRUE)

plot_data=a$counts
#rownames(plot_data)=a$mids
pdf(paste(filename,"pdf",sep="."),13)
barplot(plot_data,names.arg = a$mids)
dev.off()
#sorted_data=sort(data[,1])
#write.table(x=sorted_data,file =paste(filename,".sorted.tsv"),col.names = F, row.names = F)
