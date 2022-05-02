#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include <tuple>
//#include <unistd.h>
#include <ctime>

//Fe2 + H2O2 -> Fe3 + OH·
#define kP1 kFixed[4]
//Fe3 + hv -> Fe2s + OH·
#define kP2 kFixed[1]
//Fe3 + hv -> Fe2 + OH·
#define kP4 kFixed[3]
//Fe2s + H2O2 -> Fe3 +OH·
#define kP6 kFixed[0]

//Morg+OH· -> nada
#define kP3 kFixed[2]


//B(i) + OH· -> B(i-1)
#define kB1 kFixed[10]
//B(i-1) -> B(i)
#define kB2 kFixed[11]
//Nº ataques
#define nAt kFixed[12]

//0: Perox, 1:Fe2, 2: Fe2s, 3:BN...

using namespace std;

static int loadData(vector<double> * kFixed, vector<int> * jumpPos, double * FeValues, double * GValues,
                    vector<vector<int>> *bactTimes, vector<vector<double>> *bactConc, vector<vector<int>> *peroxTimes,
                    vector<vector<double>> *peroxConc)
{
    unsigned t0, t1;
    t0=clock();
    //Recibe en args las constantes que no tiene que guardar porque van a ser optimizadas
    printf("Loading data curvas\n");

    FILE * readingFile;
    char * readedLine = (char*)malloc(sizeof(char)*100);
    int kPos = 0;

	//leer las posiciones a saltar
	readingFile = fopen("src/environments/fotocaos_perox_model/posiciones.txt","r");
	while (fgets(readedLine,100,readingFile))
	{
		jumpPos->push_back(atoi(readedLine));
	}
	fclose(readingFile);

	//leer las constantes fijas (valor)
	readingFile = fopen("src/environments/fotocaos_perox_model/logAlphaValues.txt","r");
	while (fgets(readedLine,100,readingFile))
	{
		kFixed->push_back(atof(readedLine));
	}
	fclose(readingFile);

	for(int i = 0; i < kFixed->size(); i++)
		if(i!=12)
			kFixed->at(i)=pow(10.0, kFixed->at(i));

//    for (int i=0; i < kFixed->size(); i++)
//	{
//	    printf("load kFixed: %.5f\n", kFixed->at(i));
//	}

	//leer hierro en FeValues
	readingFile = fopen("src/environments/fotocaos_perox_model/Fe.txt", "r");
	kPos = 0;
	while (fgets(readedLine,100,readingFile))
	{
		FeValues[kPos]=atof(readedLine);
		kPos++;
	}

	fclose(readingFile);

	//leer radiacion incidente en GValues
	readingFile = fopen("src/environments/fotocaos_perox_model/ExpEa.txt", "r");
	kPos = 0;
	while (fgets(readedLine,100,readingFile))
	{
		GValues[kPos]=atof(readedLine);
		kPos++;
	}
	fclose(readingFile);

	vector<int> currentTimes;
	vector<double> currentBacts;
	for(int i_exp = 0; i_exp < 12; i_exp++)
	{
		string fileName = "src/environments/fotocaos_perox_model/Exp" + to_string(i_exp+1);
		fileName += "/DatosExp1.txt";
		readingFile = fopen(fileName.c_str(), "r");

		while (fgets(readedLine,100,readingFile))
		{
			string readedString(readedLine);
			if(readedString.length() < 3)
				break;
			int nPos = readedString.find_first_not_of("-.0123456789");
			currentTimes.push_back(atoi(readedString.substr(0,nPos).c_str())*60);
			currentBacts.push_back(atof(readedString.substr(nPos).c_str())*1000.0/6.022e23);
		}
		fclose(readingFile);
		bactTimes->push_back(currentTimes);
		bactConc->push_back(currentBacts);

		currentTimes.clear();
		currentTimes.shrink_to_fit();
		currentBacts.clear();
		currentBacts.shrink_to_fit();
	}

	vector<double> currentPerox;
	for(int i_exp = 0; i_exp < 12; i_exp++)
	{
		string fileName = "src/environments/fotocaos_perox_model/DatosPerox/DatosPerox" + to_string(i_exp+1) + ".txt";
		readingFile = fopen(fileName.c_str(), "r");

		while (fgets(readedLine,100,readingFile))
		{
			string readedString(readedLine);
			if(readedString.length() < 3)
				break;
			int nPos = readedString.find_first_not_of("-.0123456789");
			currentTimes.push_back(atoi(readedString.substr(0,nPos).c_str())*60);
			currentPerox.push_back(atof(readedString.substr(nPos).c_str())* 0.001 / 34.0147);
		}
		fclose(readingFile);
		peroxTimes->push_back(currentTimes);
		peroxConc->push_back(currentPerox);
		currentTimes.clear();
		currentTimes.shrink_to_fit();
		currentPerox.clear();
		currentPerox.shrink_to_fit();
	}

	readingFile = fopen("src/environments/fotocaos_perox_model/deltaTime.txt", "r");
	fgets(readedLine,100,readingFile);
	int timeSteps = atoi(readedLine);
	fclose(readingFile);

    t1 = clock();
    double time = (double(t1-t0)/CLOCKS_PER_SEC);
    printf("Model loading time: %.3f\n", time);
    return timeSteps;
}

static vector<double> kFixed;
static vector<int> jumpPos;
static double * FeValues = (double*)malloc(sizeof(double)*12);
static double * GValues = (double*)malloc(sizeof(double)*12);
static vector<vector<int>> bactTimes;
static vector<vector<double>> bactConc;
static vector<vector<int>> peroxTimes;
static vector<vector<double>> peroxConc;
static int timeSteps = loadData(&kFixed, &jumpPos, FeValues, GValues, &bactTimes, &bactConc, &peroxTimes, &peroxConc);
static const int rows = 13;
static const int colums = 7201;
static double * curvas = (double *)malloc((rows * colums + colums) * sizeof(double));
static double error_glob = 0.0;


extern "C" double * generarCurvas(double * param, int nParam)
{
    unsigned t0, t1;
    t0=clock();

	for (int i = 0; i < nParam; i++)
	{
//	    printf("param: %.3f, jumpPos: %d, kFixed: %.3f\n", param[i], jumpPos[i], kFixed[jumpPos[i]]);
        kFixed[jumpPos[i]] = pow(10.0, param[i]);
	}

//     vector< vector<int> >::iterator row;
//     vector<int>::iterator col;
//     for (row = peroxTimes.begin(); row != peroxTimes.end(); row++) {
//         for (col = row->begin(); col != row->end(); col++) {
//             printf("PeroxTimes: %d\n", *col);
//         }
//     }
//
//     for (row = peroxTimes.begin(); row != peroxTimes.end(); row++) {
//         for (col = row->begin(); col != row->end(); col++) {
//             printf("bactTimes: %d\n", *col);
//         }
//     }

	double error = 0.0;
	double OH = 0.0;
	int N = floor(nAt);
	vector<double> cOld(N + 3);
	vector<double> cNew(N + 3);
	double At = 1.0/(double)timeSteps;

    vector<int> combinedTimes;
	vector<int> owner;

//    int rows = 12;
//    int colums = 7201;

//    double * curvas = (double *)malloc((rows * colums + colums) * sizeof(double));
    int cont_curv = 0;

	for(int i_exp = 0; i_exp < 12; i_exp++)
	{
// 	    printf("Experiment: %d\n", i_exp);
//		string fileName = "DatosPerox" + to_string(i_exp+1) + ".txt";
//		writingFile = fopen(fileName.c_str(), "w");
		for(int i = 0; i < (N + 3); i++)
		{
			cOld[i]=0.0;
			cNew[i]=0.0;
		}
		cNew[0] = peroxConc[i_exp][0];
		cNew[1] = FeValues[i_exp];
		double ViabB = bactConc[i_exp][0];
		cNew[3] = ViabB;
		int bData = bactTimes[i_exp].size();
		int pData = peroxTimes[i_exp].size();
		int lastTime = 0;
		double G = GValues[i_exp];
		double FeTot = FeValues[i_exp];
		double rProd = kP1*cNew[0]*cNew[1]+kP6*cNew[2]*cNew[0]+(kP2+kP4)*G*(FeTot-cNew[1]-cNew[2]);
		double rConsum = kP3+kB1*ViabB;
		//OH = (sqrt(rConsum*rConsum+8.0*kP4*rProd)-rConsum)/(4.0*kP4);
		OH =rProd/rConsum;
//		fprintf(writingFile, "%d\t%g\n", 0, cNew[0]*34.0147/0.001);

        int peroxPos = 0;
		int bactPos = 0;

		/// ******************************** Guardar valor en curvas ******************************************
		int lastCombTimeIndex = 1;
        curvas[cont_curv] = cNew[0]*34.0147/0.001;
        cont_curv++;

        // Añado el -1 por que está accediendo a una posisión de memoria fuera de los vectores
        while((peroxPos < pData) && (bactPos < bData))
		{
			if (peroxTimes[i_exp][peroxPos] == bactTimes[i_exp][bactPos])
			{
				combinedTimes.push_back(peroxTimes[i_exp][peroxPos]);
				owner.push_back(2);
				peroxPos++;
				bactPos++;
				continue;
			}
			if(peroxTimes[i_exp][peroxPos] < bactTimes[i_exp][bactPos])
			{
				combinedTimes.push_back(peroxTimes[i_exp][peroxPos]);
				owner.push_back(0);
				peroxPos++;
				continue;
			}
			combinedTimes.push_back(bactTimes[i_exp][bactPos]);
			owner.push_back(1);
			bactPos++;
		}

// 		for (int i = 0; i < combinedTimes.size(); i++)
// 		{
// 		    printf("Rellenar owner: %d\n", owner[i]);
// 		}


		peroxPos = 0;
		bactPos = 0;

		for(int t = 1; t < 7201; t++)
		{
			for(int subt = 0; subt < timeSteps; subt++)
			{
				for(int i = 0; i < (N+3); i++)
					cOld[i]=cNew[i];
				cNew[0] = cOld[0]+At*(-kP1*cOld[0]*cOld[1]-kP6*cOld[0]*cOld[2]);
				cNew[1] = cOld[1]+At*(kP4*G*(FeTot-cOld[1]-cOld[2])-kP1*cOld[0]*cOld[1]);
				cNew[2] = cOld[2]+At*(kP2*G*(FeTot-cOld[1]-cOld[2])-kP6*cOld[0]*cOld[2]);
				cNew[3] = cOld[3]+At*(kB2*cOld[4]-kB1*cOld[3]*OH);
				for(int AtLvl = 4; AtLvl < (N+2); AtLvl++)
					cNew[AtLvl] = cOld[AtLvl]+At*(kB1*OH*cOld[AtLvl-1]+kB2*cOld[AtLvl+1]-(kB1*OH+kB2)*cOld[AtLvl]);
				cNew[N+2] = cOld[N+2]+At*(kB1*OH*cOld[N+1]-(kB1*OH+kB2)*cOld[N+2]);
				ViabB=0.0;
				for(int AtLvl = 3; AtLvl < (N+3); AtLvl++)
					ViabB += cNew[AtLvl];
				if(ViabB<0)
					ViabB=0.0;
				rProd = kP1*cNew[0]*cNew[1]+kP6*cNew[2]*cNew[0]+(kP2+kP4)*G*(FeTot-cNew[1]-cNew[2]);
				rConsum = kP3+kB1*ViabB;
				//OH = (sqrt(rConsum*rConsum+8.0*kP4*rProd)-rConsum)/(4.0*kP4);
				OH =rProd/rConsum;
			}

			if (combinedTimes[lastCombTimeIndex] == t)
			{
// 			    printf("owner: %d, experiment Time: %d \n", owner[lastCombTimeIndex], t);
                if((owner[lastCombTimeIndex] == 0) || (owner[lastCombTimeIndex] == 2))
                {
//                     printf("CombinedTimes: %d, index: %d time: %d cNew[0]: %.5f, peroxConc: %.5f\n", combinedTimes[lastCombTimeIndex], lastCombTimeIndex, t, cNew[0], peroxConc[i_exp][peroxPos]);
                    error += 1.0e6 * pow(cNew[0]-peroxConc[i_exp][peroxPos],2.0);
                    peroxPos++;
                }
                if ((owner[lastCombTimeIndex] == 1) || (owner[lastCombTimeIndex] == 2))
                {
                    //error += 1.0e-3 * pow(log(ViabB)-log(bactConc[i_exp][bactPos]),2.0);
                    bactPos++;
                }
                lastCombTimeIndex++;
             }
//             if (t == 7200)
//             {
//                 printf("CombinedTimes: %d, index: %d time: %d combinedTimes size: %d\n", combinedTimes[lastCombTimeIndex-1], lastCombTimeIndex-1, t, (int)combinedTimes.size());
//             }
//			fprintf(writingFile, "%g\t%g\n", t/60.0, cNew[0]*34.0147/0.001);
			/// ******************************** Guardar valor en curvas ******************************************
			curvas[cont_curv] = cNew[0]*34.0147/0.001;


			cont_curv++;
		}
// 		sleep(3);
        curvas[cont_curv] = error;
		error_glob = error;
		combinedTimes.clear();
	    combinedTimes.shrink_to_fit();
	    owner.clear();
	    owner.shrink_to_fit();

//		fclose(writingFile);
	}

//	kOpt.clear();
//	kOpt.shrink_to_fit();
//	jumpPos.clear();
//	jumpPos.shrink_to_fit();
//	kFixed.clear();
//	kFixed.shrink_to_fit();
//	bactTimes.clear();
//	bactTimes.shrink_to_fit();
//	bactConc.clear();
//	bactConc.shrink_to_fit();
//	peroxTimes.clear();
//	peroxTimes.shrink_to_fit();
//	peroxConc.clear();
//	peroxConc.shrink_to_fit();
	cOld.clear();
	cOld.shrink_to_fit();
	cNew.clear();
	cNew.shrink_to_fit();
//	free(readedLine);
//	free(FeValues);
//	free(GValues);

    t1 = clock();
    double time = (double(t1-t0)/CLOCKS_PER_SEC);
//    printf("Model execution time: %.3f\n", time);
	return curvas;
}

extern "C" double cinetica(double* param, int nParam)
{
    	for (int i = 0; i < nParam; i++)
	{
//	    printf("param: %.3f, jumpPos: %d, kFixed: %.3f\n", param[i], jumpPos[i], kFixed[jumpPos[i]]);
        kFixed[jumpPos[i]] = pow(10.0, param[i]);
	}

	double error = 0.0;
	double OH = 0.0;
	int N = floor(nAt);
	vector<double> cOld(N + 3);
	vector<double> cNew(N + 3);

	vector<int> combinedTimes;
	vector<int> owner;

	double At = 1.0/(double)timeSteps;
	for(int i_exp = 0; i_exp < 12; i_exp++)
	{
		for(int i = 0; i < (N + 3); i++)
		{
			cOld[i]=0.0;
			cNew[i]=0.0;
		}
		cNew[0] = peroxConc[i_exp][0];
		cNew[1] = FeValues[i_exp];
		double ViabB = bactConc[i_exp][0];
		cNew[3] = ViabB;
		int bData = bactTimes[i_exp].size();
		int pData = peroxTimes[i_exp].size();
		int lastTime = 0;
		double G = GValues[i_exp];
		double FeTot = FeValues[i_exp];
		double rProd = kP1*cNew[0]*cNew[1]+kP6*cNew[2]*cNew[0]+(kP2+kP4)*G*(FeTot-cNew[1]-cNew[2]);
		double rConsum = kP3+kB1*ViabB;
		OH = rProd/rConsum;
		//OH = (sqrt(rConsum*rConsum+8.0*kP4*rProd)-rConsum)/(4.0*kP4);

		//Creo un vector de tiempos combinados, y un vector de enteros para saber si esos tiempos están en Perox(0), en Bact(1), o en ambos (2)
		int peroxPos = 0;
		int bactPos = 0;


		while((peroxPos < pData) && (bactPos < bData))
		{
			if (peroxTimes[i_exp][peroxPos] == bactTimes[i_exp][bactPos])
			{
				combinedTimes.push_back(peroxTimes[i_exp][peroxPos]);
				owner.push_back(2);
				peroxPos++;
				bactPos++;
				continue;
			}
			if(peroxTimes[i_exp][peroxPos] < bactTimes[i_exp][bactPos])
			{
				combinedTimes.push_back(peroxTimes[i_exp][peroxPos]);
				owner.push_back(0);
				peroxPos++;
				continue;
			}
			combinedTimes.push_back(bactTimes[i_exp][bactPos]);
			owner.push_back(1);
			bactPos++;
		}
		peroxPos = 0;
		bactPos = 0;
		int nData = combinedTimes.size();
		for(int expTime = 1; expTime < nData; expTime++)
		{
			for(int t = lastTime+1; t <= combinedTimes[expTime]; t++)
			{
				for(int subt = 0; subt < timeSteps; subt++)
				{
					for(int i = 0; i < (N+3); i++)
						cOld[i]=cNew[i];
					cNew[0] = cOld[0]+At*(-kP1*cOld[0]*cOld[1]-kP6*cOld[0]*cOld[2]);
					cNew[1] = cOld[1]+At*(kP4*G*(FeTot-cOld[1]-cOld[2])-kP1*cOld[0]*cOld[1]);
					cNew[2] = cOld[2]+At*(kP2*G*(FeTot-cOld[1]-cOld[2])-kP6*cOld[0]*cOld[2]);
					cNew[3] = cOld[3]+At*(kB2*cOld[4]-kB1*cOld[3]*OH);
					for(int AtLvl = 4; AtLvl < (N+2); AtLvl++)
						cNew[AtLvl] = cOld[AtLvl]+At*(kB1*OH*cOld[AtLvl-1]+kB2*cOld[AtLvl+1]-(kB1*OH+kB2)*cOld[AtLvl]);
					cNew[N+2] = cOld[N+2]+At*(kB1*OH*cOld[N+1]-(kB1*OH+kB2)*cOld[N+2]);
					ViabB=0.0;
					for(int AtLvl = 3; AtLvl < (N+3); AtLvl++)
						ViabB += cNew[AtLvl];
					if(ViabB<0)
						ViabB=0.0;
					rProd = kP1*cNew[0]*cNew[1]+kP6*cNew[2]*cNew[0]+(kP2+kP4)*G*(FeTot-cNew[1]-cNew[2]);
					rConsum = kP3+kB1*ViabB;
					//OH = (sqrt(rConsum*rConsum+8.0*kP4*rProd)-rConsum)/(4.0*kP4);
					OH = rProd/rConsum;
				}
			}
			lastTime=combinedTimes[expTime];
			if((owner[expTime] == 0) || (owner[expTime] == 2))
			{
				error += 1.0e6 * pow(cNew[0]-peroxConc[i_exp][peroxPos],2.0);
				peroxPos++;
			}
			if ((owner[expTime] == 1) || (owner[expTime] == 2))
			{
				//error += 1.0e-3 * pow(log(ViabB)-log(bactConc[i_exp][bactPos]),2.0);
				bactPos++;
			}
		}

	    combinedTimes.clear();
	    combinedTimes.shrink_to_fit();
	    owner.clear();
	    owner.shrink_to_fit();
	}

//	kOpt.clear();
//	kOpt.shrink_to_fit();
//	jumpPos.clear();
//	jumpPos.shrink_to_fit();
//	kFixed.clear();
//	kFixed.shrink_to_fit();
//	bactTimes.clear();
//	bactTimes.shrink_to_fit();
//	bactConc.clear();
//	bactConc.shrink_to_fit();
//	peroxTimes.clear();
//	peroxTimes.shrink_to_fit();
//	peroxConc.clear();
//	peroxConc.shrink_to_fit();
	cOld.clear();
	cOld.shrink_to_fit();
	cNew.clear();
	cNew.shrink_to_fit();
//	free(readedLine);
//	free(FeValues);
//	free(GValues);
//    free(readingFile);
    return error;
}

extern "C" double return_error()
{
return error_glob;
}


extern "C" void free_C_string(double * ptr) {
    free(ptr);
}