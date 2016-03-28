FROM ubuntu:14.04

RUN apt-get install -y postgresql

USER postgres
RUN service postgresql start && \
	psql --command "create database hypero;" && \
	psql --command "create user hypero with password 'hypero';" && \
	psql --command "grant all privileges on database hypero to hypero;"

USER root
RUN echo "host all all all trust" >> /etc/postgresql/9.3/main/pg_hba.conf
RUN echo "listen_addresses='*'" >> /etc/postgresql/9.3/main/postgresql.conf
EXPOSE 5432

RUN service postgresql restart
