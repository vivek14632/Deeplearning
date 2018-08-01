names(ImageNames_Asc)=c("Image_Name")
ImageNames_Asc$Id<-NA
ImageNames_Asc$Id<-sapply(strsplit(as.character(ImageNames_Asc$Image_Name),"_"), `[`, 1)
colnames(ImageNames_Asc)[2] <- "id"
total <- merge(ImageNames_Asc,listings.2,by="id",all.y=TRUE)