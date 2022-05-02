//g++ -fPIC -shared -o modelos3Completos.so modelos3Completos_v2.c
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

//OH· + OH· -> H2O2
#define kP5 kFixed[5]

//Morg+OH· -> nada
#define kP3 kFixed[2]

// B(i) + hv -> B(i-bla)
//#define kS kFixed[8]

//Nº de ataques Sodis
//#define nS kFixed[9]

//Bu + hv -> Bd(N)
#define kAts kFixed[6]
//Bd(x) -> Bu
#define kROH1 kFixed[7]
//
#define kROH2 kFixed[8]
//
#define kAtOH kFixed[9]
//Bu + OH· -> Bd(N)

#define kRS1 kFixed[10]
//#define kRS1 0.004786301

#define kRS2 kFixed[11]

//#define kAtS 0.000158489

//Nº ataques OH
#define nAtOH kFixed[12]

//Nº ataques Sodis
#define nAtSod kFixed[13]

#define NOBACTCOMPOUND 4
//0: Perox, 1:Fe2, 2: Fe2s, 3: OH·, 4:BN...

using namespace std;

static double * FeValues, * GValues, * GSodisValues, * bactConc, * peroxConc, * bactSodisConc, * kFixed;
static int * bactTimes, * peroxTimes, * bactSodisTimes, * bactSizes, * peroxSizes, * bactSodisSizes, * jumpPos;
static int timeSteps, optimizedVars;
static const int rows = 13;
static const int colums = 7201;
static double * curvas = (double *)malloc((rows * colums + colums) * sizeof(double));
//static double error_glob = 0.0;


extern "C" void loadData()
{
	unsigned t0, t1;
	t0 = clock();
	printf("Loading data curvas\n");
	FILE* readingFile;
	char* readedLine = (char*)malloc(sizeof(char) * 100);

	//leer hierro en FeValues
	FeValues = (double*)malloc(sizeof(double) * 12);
	readingFile = fopen("src/environments/fotocaos_complete_model/Fe.txt", "r");
	int kPos = 0;
	while (fgets(readedLine, 100, readingFile) && kPos < 12)
	{
		if (readedLine[0] == '\n')
			continue;
		FeValues[kPos] = atof(readedLine);
		kPos++;
	}
	fclose(readingFile);

	//leer energia absorbida en GValues

	GValues = (double*)malloc(sizeof(double) * 12);
	readingFile = fopen("src/environments/fotocaos_complete_model/ExpEa.txt", "r");
	kPos = 0;
	while (fgets(readedLine, 100, readingFile) && kPos < 12)
	{
		if (readedLine[0] == '\n')
			continue;
		GValues[kPos] = atof(readedLine);
		kPos++;
	}
	fclose(readingFile);

	GSodisValues = (double*)malloc(sizeof(double) * 7);
	readingFile = fopen("src/environments/fotocaos_complete_model/EaSodis.txt", "r");
	kPos = 0;
	while (fgets(readedLine, 100, readingFile) && kPos < 7)
	{
		if (readedLine[0] == '\n')
			continue;
		GSodisValues[kPos] = atof(readedLine);
		kPos++;
	}
	fclose(readingFile);

	int totalBactPoints = 0;
	bactSizes = (int*)malloc(sizeof(int) * 12);

	//leer datos experimentales de bacterias

	for (int i_exp = 0; i_exp < 12; i_exp++)
	{
		string fileName = "src/environments/fotocaos_complete_model/Exp" + to_string(i_exp + 1);
		fileName += "/DatosExp1.txt";
		readingFile = fopen(fileName.c_str(), "r");
		kPos = 0;
		while (fgets(readedLine, 100, readingFile))
		{
			if (readedLine[0] == '\n')
				continue;
			kPos++;
		}
		fclose(readingFile);
		bactSizes[i_exp] = kPos;
		totalBactPoints += kPos;
	}

	bactConc = (double*)malloc(sizeof(double) * totalBactPoints);
	bactTimes = (int*)malloc(sizeof(int) * totalBactPoints);
	kPos = 0;
	for (int i_exp = 0; i_exp < 12; i_exp++)
	{
		string fileName = "src/environments/fotocaos_complete_model/Exp" + to_string(i_exp + 1);
		fileName += "/DatosExp1.txt";
		readingFile = fopen(fileName.c_str(), "r");

		while (fgets(readedLine, 100, readingFile))
		{
			if (readedLine[0] == '\n')
				continue;
			string readedString(readedLine);

			int nPos = readedString.find_first_not_of("-.0123456789");
			bactTimes[kPos] = atoi(readedString.substr(0, nPos).c_str()) * 60;
			bactConc[kPos] = atof(readedString.substr(nPos).c_str()) * 1000.0 / 6.022e23;
			kPos++;
		}
		fclose(readingFile);
	}


	//leer datos experimentales de peroxido
	int totalPeroxPoints = 0;
	peroxSizes = (int*)malloc(sizeof(int) * 12);
	for (int i_exp = 0; i_exp < 12; i_exp++)
	{
		string fileName = "src/environments/fotocaos_complete_model/DatosPerox/DatosPerox" + to_string(i_exp + 1) + ".txt";
		readingFile = fopen(fileName.c_str(), "r");
		kPos = 0;
		while (fgets(readedLine, 100, readingFile))
		{
			if (readedLine[0] == '\n')
				continue;
			kPos++;
		}
		fclose(readingFile);
		peroxSizes[i_exp] = kPos;
		totalPeroxPoints += kPos;
	}

	peroxConc = (double*)malloc(sizeof(double) * totalPeroxPoints);
	peroxTimes = (int*)malloc(sizeof(int) * totalPeroxPoints);
	kPos = 0;
	for (int i_exp = 0; i_exp < 12; i_exp++)
	{
		string fileName = "src/environments/fotocaos_complete_model/DatosPerox/DatosPerox" + to_string(i_exp + 1) + ".txt";
		readingFile = fopen(fileName.c_str(), "r");

		while (fgets(readedLine, 100, readingFile))
		{
			if (readedLine[0] == '\n')
				continue;
			string readedString(readedLine);
			int nPos = readedString.find_first_not_of("-.0123456789");

			peroxTimes[kPos] = atoi(readedString.substr(0, nPos).c_str()) * 60;
			peroxConc[kPos] = atof(readedString.substr(nPos).c_str()) * 0.001 / 34.0147;
			kPos++;
		}
		fclose(readingFile);
	}


	//leer datos experimentales de Sodis
	int totalSodisPoints = 0;
	bactSodisSizes = (int*)malloc(sizeof(int) * 7);
	for (int i_exp = 0; i_exp < 7; i_exp++)
	{
		string fileName = "src/environments/fotocaos_complete_model/SodisData/DatosExp" + to_string(i_exp + 1) + ".txt";
		readingFile = fopen(fileName.c_str(), "r");
		kPos = 0;
		while (fgets(readedLine, 100, readingFile))
		{
			if (readedLine[0] == '\n')
				continue;
			kPos++;
		}
		fclose(readingFile);
		bactSodisSizes[i_exp] = kPos;
		totalSodisPoints += kPos;
	}

	bactSodisConc = (double*)malloc(sizeof(double) * totalSodisPoints);
	bactSodisTimes = (int*)malloc(sizeof(int) * totalSodisPoints);
	kPos = 0;
	for (int i_exp = 0; i_exp < 7; i_exp++)
	{
		string fileName = "src/environments/fotocaos_complete_model/SodisData/DatosExp" + to_string(i_exp + 1) + ".txt";
		readingFile = fopen(fileName.c_str(), "r");

		while (fgets(readedLine, 100, readingFile))
		{
			if (readedLine[0] == '\n')
				continue;
			string readedString(readedLine);
			int nPos = readedString.find_first_not_of("-.0123456789");
			bactSodisTimes[kPos] = atoi(readedString.substr(0, nPos).c_str()) * 60;
			bactSodisConc[kPos] = atof(readedString.substr(nPos).c_str()) * 1000.0 / 6.022e23;
			kPos++;
		}
		fclose(readingFile);
	}

	//lee las constantes cinéticas y las posiciones de las constantes a optimizar
	kFixed = (double*)malloc(sizeof(double) * 14);
	optimizedVars = 0;
	readingFile = fopen("src/environments/fotocaos_complete_model/posiciones.txt", "r");
	while (fgets(readedLine, 100, readingFile))
	{
		if (readedLine[0] == '\n')
			continue;
		optimizedVars++;
	}
	fclose(readingFile);
	readingFile = fopen("src/environments/fotocaos_complete_model/posiciones.txt", "r");
	jumpPos = (int*)malloc(sizeof(int) * optimizedVars);
	kPos = 0;
	while (fgets(readedLine, 100, readingFile))
	{
		if (readedLine[0] == '\n')
			continue;
		jumpPos[kPos] = atoi(readedLine);
		kPos++;
	}
	fclose(readingFile);

	kPos = 0;
	readingFile = fopen("src/environments/fotocaos_complete_model/logAlphaValues.txt", "r");
	while (fgets(readedLine, 100, readingFile))
	{
		if (readedLine[0] == '\n')
			continue;
		kFixed[kPos] = atof(readedLine);
		kPos++;
	}
	fclose(readingFile);

	readingFile = fopen("src/environments/fotocaos_complete_model/deltaTime.txt", "r");
	fgets(readedLine, 100, readingFile);
	timeSteps = atoi(readedLine);
	fclose(readingFile);

	t1 = clock();
	double time = (double(t1 - t0) / CLOCKS_PER_SEC);
	printf("Model loading time: %.3f\n", time);
	/*for (int i = 0; i < totalBactPoints; i++)
		printf("%g\n", bactConc[i]);*/

}

double* generarCurvas(double* param, int nParam, bool peroxErr, bool bactErr)
{

	unsigned t0, t1;
	t0 = clock();

	for (int i = 0; i < nParam; i++)
	{
		//	    printf("param: %.3f, jumpPos: %d, kFixed: %.3f\n", param[i], jumpPos[i], kFixed[jumpPos[i]]);
		if (jumpPos[i] < 12)
			kFixed[jumpPos[i]] = pow(10.0, param[i]);
		else
			kFixed[jumpPos[i]] = param[i];
	}

	double error = 0.0;
	double OH = 0.0;
	int NOH = floor(nAtOH);
	int NSod = floor(nAtSod);
	vector<double> cOld(NOBACTCOMPOUND + NOH * NSod);
	vector<double> cNew(NOBACTCOMPOUND + NOH * NSod);

	double At = 1.0 / (double)timeSteps;

	//    int rows = 12;
	//    int colums = 7201;

	//    double * curvas = (double *)malloc((rows * colums + colums) * sizeof(double));
	int cont_curv = 0;

	//creo un contador para el peróxido y otro para las bacterias para saber los puntos recorridos
	int iBact = 0;
	int iPerox = 0;

    vector<int> combinedTimes;
	vector<int> owner;
	for (int i_exp = 0; i_exp < 12; i_exp++)
	{
//		printf("generar curvas perox 1 %d\n", i_exp);
		// 	    printf("Experiment: %d\n", i_exp);
		//		string fileName = "DatosPerox" + to_string(i_exp+1) + ".txt";
		//		writingFile = fopen(fileName.c_str(), "w");


		for (int i = 0; i < (NOBACTCOMPOUND + NOH * NSod); i++)
		{
			cOld[i] = 0.0;
			cNew[i] = 0.0;
		}
//		printf("generar curvas perox 2 %d\n", i_exp);
		cNew[0] = peroxConc[iPerox];
		cNew[1] = FeValues[i_exp];
		double ViabB = bactConc[iBact];
		double damBact = 0.0;
		cNew[4] = ViabB;
		int bData = bactSizes[i_exp];
		int pData = peroxSizes[i_exp];
		int lastTime = 0;
		double G = GValues[i_exp];
		double FeTot = 0.000179083 * (1 + i_exp % 2);//FeValues[i_exp];
		double rProd, rConsum;
		//OH = (sqrt(rConsum*rConsum+8.0*kP4*rProd)-rConsum)/(4.0*kP4);
//		fprintf(writingFile, "%d\t%g\n", 0, cNew[0]*34.0147/0.001);

		int peroxPos = 0;
		int bactPos = 0;

		/// ******************************** Guardar valor en curvas ******************************************
		curvas[cont_curv] = peroxErr ? cNew[0] * 34.0147 / 0.001 : ViabB * 6.022e23 / 1000.0;
		cont_curv++;

//		printf("generar curvas perox 3 %d\n", i_exp);
		// Añado el -1 por que está accediendo a una posisión de memoria fuera de los vectores
		while ((peroxPos < pData) && (bactPos < bData))
		{
			if (peroxTimes[iPerox + peroxPos] == bactTimes[iBact + bactPos])
			{
				combinedTimes.push_back(peroxTimes[iPerox + peroxPos]);
				owner.push_back(2);
				peroxPos++;
				bactPos++;
				continue;
			}
			if (peroxTimes[iPerox + peroxPos] < bactTimes[iBact + bactPos])
			{
				combinedTimes.push_back(peroxTimes[iPerox + peroxPos]);
				owner.push_back(0);
				peroxPos++;
				continue;
			}
			combinedTimes.push_back(bactTimes[iBact + bactPos]);
			owner.push_back(1);
			bactPos++;
		}
		while (peroxPos < pData)
		{
			combinedTimes.push_back(peroxTimes[iPerox + peroxPos]);
			owner.push_back(0);
			peroxPos++;
		}
		while(bactPos<bData)
		{
			combinedTimes.push_back(bactTimes[iBact + bactPos]);
			owner.push_back(1);
			bactPos++;
		}
//		printf("generar curvas perox 4 %d\n", i_exp);
		// 		for (int i = 0; i < combinedTimes.size(); i++)
		// 		{
		// 		    printf("Rellenar owner: %d\n", owner[i]);
		// 		}


		peroxPos = 0;
		bactPos = 0;

//		printf("combinedTimes: %d\n", (int)combinedTimes.size());
		for (int expTime = 1; expTime < combinedTimes.size(); expTime++)
		{
//			printf("generar curvas perox 5 %d, %d\n", i_exp, expTime);
			for (int t = lastTime + 1; t <= combinedTimes[expTime]; t++)
			{
				for (int subt = 0; subt < timeSteps; subt++)
				{
					for (int i = 0; i < (NOBACTCOMPOUND + NOH * NSod); i++)
						cOld[i] = cNew[i];
					double r1 = kP1 * cOld[0] * cOld[1];
					double r2 = kP6 * cOld[0] * cOld[2];
//					double r3 = kP4 * G * (FeTot - cOld[1] - cOld[2]);
                    double r3 = 0.0;
					double r4 = kP2 * G * (FeTot - cOld[1] - cOld[2]);
					rProd = r1 + r2 + r3 + r4;
					rConsum = kP3 + kAtOH * ViabB;
					cOld[3] = (sqrt(rConsum * rConsum + 8.0 * kP5 * rProd) - rConsum) / (4.0 * kP5);
					cNew[0] = cOld[0] + At * (-r1 - r2);
					cNew[1] = cOld[1] + At * (r3 - r1);
					cNew[2] = cOld[2] + At * (r4 - r2);
					//Primer nivel de OH
					for (int i = NOBACTCOMPOUND; i < (NOBACTCOMPOUND + NOH * NSod); i++)
					{
						//Elementos generales: ataque
						cNew[i] = cOld[i] - At * (kAtOH * cOld[3] + kAts * G) * cOld[i];
						//No han recibido luz
						if (((i - NOBACTCOMPOUND) % NSod) == 0)
						{
							//restan regeneracion OH1: sin luz, más de primer nivel
							if (i > NOBACTCOMPOUND + NSod - 1)
							{
								cNew[i] = cNew[i] - At * kROH1 * cOld[i];
							}
							//Suman regeneracion OH1: sin luz, no último nivel
							if (i < (NOBACTCOMPOUND + NOH * NSod - NSod))
							{
								cNew[i] = cNew[i] + At * (kROH1 * cOld[i + NSod]);
							}
						}
						else
						{
							//han recibido luz
							//Suman ataque sodis
							cNew[i] = cNew[i] + At * (kAts * G * cOld[i - 1]);
							//Restan regeneracion OH2 y Sodis2 si no es primer nivel de OH
							if (i > NOBACTCOMPOUND + NSod - 1)
							{
								cNew[i] = cNew[i] - At * (kROH2 + kRS2) * cOld[i];
							}
							//Suman regeneracion OH2 si no es el último nivel de OH
							if (i < (NOBACTCOMPOUND + NOH * NSod - NSod))
							{
								cNew[i] = cNew[i] + At * (kROH2 * cOld[i + NSod]);
							}
						}
						//Regeneracion Sodis1: no han recibido OH, salvo el último nivel de SODIS
						if (i < NOBACTCOMPOUND + NSod - 1)
						{
							cNew[i] = cNew[i] + At * (kRS1 * cOld[i + 1]);
						}
						if (i > NOBACTCOMPOUND + NSod - 1)
						{
							cNew[i] = cNew[i] + At * (kAtOH * cOld[3] * cOld[i - NSod]);
							if (((i + 1 -NOBACTCOMPOUND) % NSod) > 0)
							{
								cNew[i] = cNew[i] + At * (kRS2 * cOld[i + 1]);
							}
						}
						if (i > NOBACTCOMPOUND && i < NOBACTCOMPOUND + NSod)
						{
							cNew[i] = cNew[i] - At * kRS1 * cOld[i];
						}
					}
					ViabB = 0.0;
					for (int i = NOBACTCOMPOUND; i < (NOBACTCOMPOUND + NOH * NSod); i++)
					{

						ViabB += cNew[i];
					}
					if (ViabB < 0.0)
						ViabB = 0.0;

					//OH = ;
					//OH = rProd/rConsum;
				}
				/// ******************************** Guardar valor en curvas ******************************************
				curvas[cont_curv] = peroxErr ? cNew[0] * 34.0147 / 0.001 : ViabB * 6.022e23 / 1000.0;
				cont_curv++;
			}

			lastTime = combinedTimes[expTime];
			if ((owner[expTime] == 0) || (owner[expTime] == 2))
			{
				if (peroxErr)
					error += 3.0e5 * pow(cNew[0] - peroxConc[i_exp + iPerox], 2.0);
				peroxPos++;
			}
			if ((owner[expTime] == 1) || (owner[expTime] == 2))
			{
				if (bactErr)
					error += 1.25e-4 * G * pow(log(ViabB) - log(bactConc[i_exp + iBact]), 2.0);
				bactPos++;
			}

		}

//		printf("generar curvas perox 6\n");
//		for (int t = 0; t < combinedTimes.size(); t++) {
//			printf("%d\t", combinedTimes[t]);
//		}
		for (int t = lastTime + 1; t < 7201; t++)
		{
			//		    printf("generar curvas perox 7 %d, %d, %d\n", i_exp, t, cont_curv);
			curvas[cont_curv] = 0.0;
			cont_curv++;
		}
//		printf("generar curvas perox 8\n");

//		error_glob += error;
		iPerox += peroxSizes[i_exp];
		iBact += bactSizes[i_exp];

		// Liberar espacio
        combinedTimes.clear();
        combinedTimes.shrink_to_fit();
        owner.clear();
        owner.shrink_to_fit();
	}

	curvas[cont_curv] = error;
//	printf("generar curvas perox 9\n");

    // Liberar espacio
    cOld.clear();
	cOld.shrink_to_fit();
	cNew.clear();
	cNew.shrink_to_fit();

	t1 = clock();
	double time = (double(t1 - t0) / CLOCKS_PER_SEC);
	//    printf("Model execution time: %.3f\n", time);
	return curvas;
}

extern "C" double* generarCurvasSodis(double* param, int nParam)
{
//	printf("generar curvas sodis 1");
	unsigned t0, t1;
	t0 = clock();

	for (int i = 0; i < nParam; i++)
	{
		//	    printf("param: %.3f, jumpPos: %d, kFixed: %.3f\n", param[i], jumpPos[i], kFixed[jumpPos[i]]);
		if (jumpPos[i] < 12)
			kFixed[jumpPos[i]] = pow(10.0, param[i]);
		else
			kFixed[jumpPos[i]] = param[i];
	}
//	printf("generar curvas sodis 2");
	double error = 0.0;
	int NSod = floor(nAtSod);
	vector<double> cOld(NOBACTCOMPOUND + NSod);
	vector<double> cNew(NOBACTCOMPOUND + NSod);

	double At = 1.0 / (double)timeSteps;



	//    int rows = 12;
	//    int colums = 7201;

	//    double * curvas = (double *)malloc((rows * colums + colums) * sizeof(double));
	int cont_curv = 0;

	//creo un contador para las bacterias para saber los puntos recorridos
//	printf("generar curvas sodis 3");
	int iBact = 0;
	for (int i_exp = 0; i_exp < 7; i_exp++)
	{
		// 	    printf("Experiment: %d\n", i_exp);
		//		string fileName = "DatosPerox" + to_string(i_exp+1) + ".txt";
		//		writingFile = fopen(fileName.c_str(), "w");
		for (int i = 0; i < (NOBACTCOMPOUND + NSod); i++)
		{
			cOld[i] = 0.0;
			cNew[i] = 0.0;
		}
		double ViabB = bactSodisConc[iBact];
		double damBact = 0.0;
		cNew[4] = ViabB;
		int bData = bactSodisSizes[i_exp];
		int lastTime = 0;
		double G = GSodisValues[i_exp];

		/// ******************************** Guardar valor en curvas ******************************************
		curvas[cont_curv] = ViabB * 6.022e23 / 1000.0;
		cont_curv++;

		for (int expTime = 1; expTime < bData; expTime++)
		{
			for (int t = lastTime + 1; t <= bactSodisTimes[expTime]; t++)
			{
				for (int subt = 0; subt < timeSteps; subt++)
				{
					for (int i = 0; i < (NOBACTCOMPOUND + NSod); i++)
						cOld[i] = cNew[i];
					ViabB = 0.0;
					for (int i = NOBACTCOMPOUND; i < (NOBACTCOMPOUND + NSod); i++)
					{
						cNew[i] = cOld[i] - At * (kAts * G * cOld[i]);
						if (i > NOBACTCOMPOUND)
							cNew[i] = cNew[i] + At * (kAts * G * cOld[i - 1] - kRS1 * cOld[i]);
						if (i < (NOBACTCOMPOUND + NSod - 1))
							cNew[i] = cNew[i] + At * (kRS1 * cOld[i + 1]);
						ViabB += cNew[i];
					}
				}
				/// ******************************** Guardar valor en curvas ******************************************
				curvas[cont_curv] = ViabB * 6.022e23 / 1000.0;
				cont_curv++;
			}
			error += pow(log(ViabB) - log(bactSodisConc[expTime]), 2.0);
			lastTime = bactSodisTimes[expTime];
		}
		for (int t = lastTime + 1; t < 7201; t++)
		{
			curvas[cont_curv] = 0.0;
			cont_curv++;
		}
		iBact += bData;
	}
	curvas[cont_curv] = error;
//	error_glob = error;
//	printf("generar curvas sodis 4");

    // Liberar espacio
    cOld.clear();
	cOld.shrink_to_fit();
	cNew.clear();
	cNew.shrink_to_fit();

	t1 = clock();
	double time = (double(t1 - t0) / CLOCKS_PER_SEC);
	//    printf("Model execution time: %.3f\n", time);
	return curvas;
}

extern "C" double* generarCurvasPerox(double* param, int nParam)
{
	return generarCurvas(param, nParam, true, false);
}
extern "C" double* generarCurvasBact(double* param, int nParam)
{
	return generarCurvas(param, nParam, false, true);
}

//extern "C" double return_error()
//{
//	return error_glob;
//}


extern "C" void free_Arrays() {
	free(FeValues);
	free(GValues);
	free(GSodisValues);
	free(bactConc);
	free(peroxConc);
	free(bactSodisConc);
	free(kFixed);
	free(jumpPos);
	free(bactTimes);
	free(peroxTimes);
	free(bactSodisTimes);
	free(bactSizes);
	free(peroxSizes);
	free(bactSodisSizes);
	free(curvas);
}
