% MODULO DE REGRESION NO LINEAL POR MINIMIZACIÓN DE ERROR
% =======================================================
clear

global Fe_exp;
global e_a_exp;


strDir = pwd();
strDir2 = substr(strDir,length(strDir));
numeroExperimento = str2num(strDir2);
Fe = load('../Fe.txt');
e_a = load('../ExpEa.txt');
Fe_exp = Fe(numeroExperimento,2)
e_a_exp = e_a(numeroExperimento,2)

% Valores iniciales de los parametros cinéticos e intervalos LB < alpha < UB:
logalpha_ini = load('..\ValoresAlpha.txt');

DatosExp1 = load('DatosExp1.txt');
factorFromUFCToMol = 1000/(6.022e+23);
DatosPerox = load(strcat('../DatosPerox/DatosPerox',strDir2,'.txt'));
%DatosExp2 = load('DatosExp2.txt'); 


% Condiciones iniciales 
% C0 = [CA0 CB0 CC0 ...] mol L-1
global C_0; 
%C_0 = load("C_0.txt");
CA_0 = DatosExp1(1,2)*factorFromUFCToMol;
CB_0 = 0;
CC_0 = DatosPerox(1,2)*0.001/(34.0147);  %mG/l --> mol/l%


logalpha = logalpha_ini

% SIMULAR MODELO Y REPRESENTAR AJUSTE:
%=====================================

% Calcular modelo hasta equilibrio

A1 = (10^logalpha(1));
Coeff = 166511.8959; %L mol-1 m-1
K1 = 70 ;   % M-1 s-1
%En esta primera etapa, la resolución del hierro en equilibrio requiere una ecuación de segundo grado,
% para considerar el cambio en la concentración de Peróxido, que es importante
eqSecOrderA = K1;
eqSecOrderB = A1*Coeff*e_a_exp + K1*CC_0 - K1*Fe_exp;
eqSecOrderC = -A1* Fe_exp*Coeff* e_a_exp;
eqFe = (-eqSecOrderB + sqrt(eqSecOrderB*eqSecOrderB - 4*eqSecOrderA*eqSecOrderC) ) / (2*eqSecOrderA);

eqM = CC_0 / Fe_exp;
eqXa = (Fe_exp - eqFe) / Fe_exp;
t_0 = log( (eqM - eqXa) / (eqM * (1-eqXa)) ) / (K1 * (CC_0 - Fe_exp) );
CC_0 = CC_0 - (Fe_exp - eqFe);

C_0 = [CA_0 CB_0 CC_0];

% Calcular modelo desde equilibrio

%t_0 = 0;  
t_f = DatosPerox(length(DatosPerox),1)*60;  % Valor del tiempo a calcular;
ptos = 1000;
% Resolución de la cinética
addpath('../');
[t,C]=feuler('cinetica', [t_0,t_f], C_0, ptos,logalpha);

tGraph = zeros(1,length(t)+1);
cGraph = zeros(1,length(tGraph));
tGraph(1,1) = 0;
cGraph(1,1) = DatosPerox(1,2);
for i_graph = 2 : length(tGraph)
	tGraph(1,i_graph)=t(i_graph-1)/60;
	cGraph(1,i_graph) = max(1,C(i_graph-1,3)*34.0147/0.001);

end

%Representa Datos Experimentales

%DatosExp1 = load('DatosExp1.txt'); 
hold all
semilogy(DatosPerox(:,1),DatosPerox(:,2), 'rs', 'MarkerEdgeColor','r', 'MarkerFaceColor','r', 'MarkerSize',8);
xlabel('Tiempo')
ylabel('Concentracion')

% Figura Datos Modelo

semilogy(tGraph, cGraph, '-r', 'LineWidth',2);
%plot(t, C(:,2)/CA_0, '-b', 'LineWidth',2)


% Guardar resultados
salida = [t' C];
save -ascii resultadosPerox.txt salida

