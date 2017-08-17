
/*-----------------------------------------------
 * lowrank.h
 *
 *-----------------------------------------------*/

#ifndef LOWRANKSVD_H_
#define LOWRANKSVD_H_

#include <armadillo>
#include <cmath>

using namespace arma;

class LowRankSVD {

//===============================================================================
// incSVDUpdate
//
// Initial right space basis V'= IdentityMatrix size K
// Thin QR factorization of A=QR  U=Q. The SVD of R produces the singular values of A. (C. Baker 2010)
// A=[a1;a2;a3;a4;..an] Q=[q1;q2;q3;q4;...qn] both mx1 matrix
// Rank 1 modification
//
//   X = U*S*V'    X is p x q
//
//   P is orthogonal basis of column of the column-sapce (I-UU')A, Ra=P'(I-UU')A
//   [U A] = [U P][IU' A; 0 Ra] Similarly - QRb = (I-VV')B
//
//   X + AB' = [U P]K[V Q]'
//
// Brand "incremental singular value decomposition of uncertain data missing values" - 2002
// Brand, "Fast low-rank modifications of the thin singular value decomposition".
//===============================================================================
public:
	static void incSVDUpdate(mat& U, mat& S, mat& V, mat& frame){

    // Gram-Schmidt orthogonalization row version
    mat  m = U.t()*frame;
    mat p = frame - (U*m);
    double pnorm = sqrt(accu( p % p));
    mat P;
    P.zeros(frame.n_rows, frame.n_cols);
    if( pnorm > 1e-7 ) P = p/pnorm;

    mat K =   {S[0],0,m[0],pnorm};  // C++11 only
    K.reshape(2,2);
    mat Gu, Gv;
    vec St;
    svd(Gu,St,Gv,K,"std");

    S  = St[0];
    U  = U * Gu(0,0) + P * Gu(1,0);
    V =  V * Gv(0,0);
	int nRows = V.n_rows;
 	V.resize(V.n_rows+1,1);
    V(nRows) =  Gv(1,0);
    return;
}

//===============================================================================
// incSVDDowndate
//
// At each iteration the rank of the SVD is increased by one.
// To keep the solution rank-1 the singular vectors corresponding to the smallest
// singular value if S are dropped at each iteration. However, each iteration will
// still add a new row to V corresponding to the new column. But we cannot just drop
// the old row since we need to maintain orthogonality. Instead DOWNDATE.
//===============================================================================
public:
	static void  incSVDDowndate(mat& U, mat& S, mat& V){

    int N =  V.n_cols;

    // n = V'b;   q = b-Vn; Rb = ||q|| Q = Rbinv*q in this case b is 1
    double n =  V(0);
    mat q = -V*n;     // q = b-Vn
    q(0) = q(0)+1;    //b is only 1 for the first row all others are zero
    double rho = sqrt(( 1 - n*n) );

	mat Q  = zeros<mat>(q.n_rows,1);
    if( rho > 1e-7 )  Q = q/rho;

    // armadillo reshapes rows first
    mat K =   {S[0]-S[0]*n*n,0,-rho*S[0]*n,0};  // C++11 only
    K.reshape(2,2);

    mat Gu, Gv;
    vec St;
    svd(Gu,St,Gv,K,"std");

    S  = St[0];
    U  = U * Gu(0,0);

    mat Vp  = V*Gv(0,0) + Q*Gv(1,0);

    V = Vp;
    V.shed_row(0);
	}


};

#endif
