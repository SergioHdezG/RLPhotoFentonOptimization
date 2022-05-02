function Error = Objetivo(logalpha)

global e_a_PAR;
global A2_Par;

errores = zeros (1,1);

for i_exp=1:2

    % Cargar Datos Experimentales:
    % Columna1: t 
    % Columna2: C_A 
    
	fichero1 = strcat('Exp',int2str(i_exp),'\DatosExp1.txt');
	%fichero2 = strcat('Exp',int2str(i_exp),'\DatosExp2.txt');
	
	ExpA=load(fichero1);
% ExpB=load(fichero2);
	
	A2_Par = e_a_PAR(i_exp,2);
  CT = CT_exp(i_exp,2);
	CA_0 = ExpA(1,2);
	CB_0 = 0;
%	CB_0 = ExpB(1,2);
	% C_0 = [CA_0 CB_0];

	
    % Calcular Datos Te�ricos:    
    CalcA = zeros(size(ExpA));  % Inicia matriz de valores al tama�o adecuado
    CalcB = zeros(size(ExpA));
	CalcC = zeros(size(ExpA));
    CalcA(:,1) = ExpA(:,1);     % Iguala columnas de tiempo
  	CalcB(:,1) = ExpA(:,1);
    CalcC(:,1) = ExpA(:,1);
    
	CalcA(1,2) = CA_0; 
	CalcB(1,2) = CB_0;
	CalcC(1,2) = CC_0;
	
	
	for i=2:length(CalcA)
       
	   % Tiempo de integraci�n
        t_0 = CalcA(i-1,1)*60;  
        t_f = CalcA(i,1)*60;  % Valor del tiempo a calcular;
        ptos = 50;
		
        % Resoluci�n de la cin�tica
		C_last = [CalcA(i-1,2) CalcB(i-1,2) CalcC(i-1,2)];
        [t,C]=feuler('cinetica', [t_0,t_f], C_last, ptos,logalpha);
        CalcA(i,2) = C(length(C),1);   % C_A 
        CalcB(i,2) = C(length(C),2);   % C_B 
        CalcC(i,2) = C(length(C),3);   % C_C 
		
		
    end	
     
	  save('DatosCalc1A.txt', '-ascii', 'CalcA');
    save('DatosCalc1B.txt', '-ascii', 'CalcB'); 	
	
end
	
Error = sum(errores)* 1e6	
	
	
	
end 

