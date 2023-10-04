# Big-Data-Analysis
Cyber Security for Big Data

Project Cyberitis - Automated Cyber Security System designed for Big Data Environments, Offline IoT Devices, and Complex Data Analytics

Next-generation Cyber Security System using Machine Learning, Automation, and Big Data Analytic stacks.

Project Cyberitis Big Data components consists of Hadoop, Spark and Storm based tools for outlier and anomaly detection, which are interweaved into the system's machine learning and automation engine for real-time fraud detection to forensics, intrusion detection to forensics.

Cyberitis uses the Ophidia Analytics Framework for independent Big Data Multi-Inspection / Forensics of high-level threats or volume datasets exceeding local resources. Ophidia Analytics Framework is an open source big data analytics framework, that includes parallel operators for data analysis and mining (subsetting, reduction, metadata processing, etc.) that can run over a cluster. The framework is fully integrated with Ophidia Server: it receives commands from the server and sends back notifications so that workflows can be executed efficiently.

The Cyber Security System additionally incorporates Lumify, an open source big data analysis and visualization platform to provide big data analysis and visualization of each instances of fraud or intrusion events into temporary, compartmentalized virtual machines that creates a full snapshot of the network infrastructure and infected device to allow an in-depth analytics, forensic review, and provide a transportable threat analysis for Executive level next-steps.

Cyberitis uses local and cloud resources to launch Lumify for big data analysis and visulalization (customizable per environment and user). Open Source Lumify Dev Virtual Machine includes only the backend servers (Hadoop, Accumulo, Elasticsearch, RabbitMQ, Zookeeper) used for development. This VM makes it easy for developers to get started without needing to install the full stack on thier develoment machines.

Big Data Analytics and Cyber Security Reports:


•	Simple to use
•	Input output in CSV format
•	Metadata defined in simple JSON file
•	Extremely configurable with tons of configuration knobs
Project Cyberities Additional Has Manual Tools For Big Data Analytics And Forensics:
•	Fast and flexible synthetic machine learning data generator with high degree veracity
•	Intrusion detection representative workloads
•	A user-friendly interface to monitor the cluster performance, showing application and 	system metrics
MAIN SYSTEM ARCHITECTURE
The Cyberitis System is Composed of Five Main Components:

Machine Learning Dataset Generator
Global and Self-Generating Local Representative Workloads
Independent Big Data Analytics Framework for Data Analysis and Mining
Machine Learning Metrics Of Interest And Visualization
Backend (Local and Cloud Deployable) Big Data Analysis and Visualization Platform

Component 1 & 2 Algorithms

Multi variate instance distribution model
Multi variate sequence or multi gram distribution model
Average instance Distance
Relative instance Density
Markov chain with sequence data
Instance clustering
Sequence clustering
Getting started
Component 1 & 2 Build
For Hadoop 1

mvn clean install
For Hadoop 2 (non yarn)

git checkout nuovo
mvn clean install
For Hadoop 2 (yarn)

git checkout nuovo
mvn clean install -P yarn
For Spark

mvn clean install
sbt publishLocal
in ./spark sbt clean package
