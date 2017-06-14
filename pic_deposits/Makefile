CC = gcc

#For going fast:
CCFLAGS = -shared -fPIC -O3 -Wall

SOURCES = pic_deposits.c

pic_deposits.so : $(SOURCES)
	$(CC) $(CCFLAGS) $(SOURCES) -o $@

clean :
	rm pic_deposits.so
