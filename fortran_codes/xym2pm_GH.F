! 
! 12/02/2020 -> TO DO: Post-SM4 GD correction
!

! ML -> Don't touch them unless you know what to you are doing...
#define _NVOTE_        500
#define _NSTMAX_    999999
#define _NMATMX_    999999
#define _HASHDIM_     5000
#define _HASHDEP_        7
      
      program xym2pm_Gaia
      implicit none

      real*8  RA_CEN
      real*8  DE_CEN

      real*8,    dimension(:), allocatable :: xgaia, ygaia, mgaia
      real*8,    dimension(:), allocatable :: xhst, yhst, mhst, qhst
      real*8,    dimension(:), allocatable :: xchsto, ychsto, mhsto
      real*8,    dimension(:), allocatable :: xhst_amp, yhst_amp
      real*8,    dimension(:), allocatable :: xhst_loc, yhst_loc
      real*8,    dimension(:), allocatable :: xrhsto, yrhsto
      real*8,    dimension(:), allocatable :: x1c, y1c, m1c, dmc
      real*8,    dimension(:), allocatable :: x1co, y1co, m1co, dhsto
      real*8,    dimension(:), allocatable :: dr, aux, rhst, rhsto
      real*8,    dimension(:), allocatable :: x2c, y2c, m2c, qhsto
      real*8,    dimension(:), allocatable :: x2co, y2co, m2co
      integer,   dimension(:), allocatable :: nhsto, nhst, khsto, hst_id
      integer,   dimension(:), allocatable :: n1match, n2match
      integer*8, dimension(:), allocatable :: gaia_idmat, star_id
      integer,   dimension(:), allocatable :: ngaiao, ngaia, usegaia
      logical,   dimension(:), allocatable :: okflag
      real*8 bar, sig, xglob1, yglob1
      integer NREF , NREFs, NIDGAIAMAT
      integer NSTAR, NSTARs, NMAT, NMATs, NMATu, NSTARu, NMATso
      integer NTRY
      integer NREJX
      integer NREJY
      integer NREJM
      integer NREJQ
      
      real*8 AG,BG,CG,DG,Xo,Yo,Wo,Zo
      real*8 GA,GB,GC,GD, DELTA
      real*8 XUSE,YUSE
      real*8 AGR,BGR,CGR,DGR
      real*8 GAR,GBR,GCR,GDR
      real*8 atan2dj

      real*8,    dimension(:), allocatable :: AG_K, BG_K, CG_K, DG_K
      real*8,    dimension(:), allocatable :: Xo_K, Yo_K, Wo_K, Zo_K
      real*8,    dimension(:), allocatable ::   dra_o, dde_o
      real*8,    dimension(:), allocatable ::     x_o, ex_o
      real*8,    dimension(:), allocatable ::     y_o, ey_o
      real*8,    dimension(:), allocatable ::     xpm_o, expm_o
      real*8,    dimension(:), allocatable ::     ypm_o, eypm_o
      real*8,    dimension(:), allocatable ::    ra_o, ragaia
      real*8,    dimension(:), allocatable ::    de_o, degaia
      real*8,    dimension(:), allocatable ::   era_o
      real*8,    dimension(:), allocatable ::   ede_o
      real*8,    dimension(:), allocatable ::    mg_o
      real*8,    dimension(:), allocatable ::  mura_o
      real*8,    dimension(:), allocatable :: emura_o
      real*8,    dimension(:), allocatable ::  mude_o
      real*8,    dimension(:), allocatable :: emude_o

      integer Os, Ou

      character*200 INSTRING
      character*80 STRING_PROG
      character*40 STRING_RA, STRING_RAo
      character*40 STRING_DE, STRING_DEo
      character*40 STRING_RAD
      character*20 STRING_DIAG
      character*300 STRING80
      character*80 FILERDML

      logical RA_SEXIG
      logical DE_SEXIG
      logical DE_NEG
      logical USESAT

      real RAh, RAm, RAs
      real DEd, DEm, DEs

      integer i, k
      integer O
      real*8 dra, dde, ddd
      real*8 rd2x, rd2y

      real*8 XY_XCEN, XY_YCEN, PI_314
      real*8 RD_RCEN, RD_DCEN, XY_XCEN_LOC, XY_YCEN_LOC
      real*8 RD_RCEN_LOC, RD_DCEN_LOC, RAD_DET
      real*8 XY_PSCL
      real*8 XY_RDAT, GAIA_RDAT
      real*8 aux1, aux2, aux3, aux4, aux5
      
      real*8 dYEARS
      real*8 RA_RDATE
      real*8 DE_RDATE
      real*8 XFRAME
      real*8 YFRAME
 
      logical RES

      real*8  RA1, RA2, DE1, DE2
      logical DIAG

      integer NARG, NARGs
 
      real*8 dclip

      integer L, Ls
      integer*8 idu
      integer*4 r_l
      integer*4 r_id


! ML
      logical OBJECT
      character*40 OBJECTNAME
      character*300 COMMAND
      character*300 IMGFILE, XYMFILE, MATFILE, LNKFILE, GAIAFILE
      character*300 OUTDIR

      character*20 INSTRUME
      character*20 DETECTOR
      real*8 RAREF, DECREF, XREF, YREF, RD_RAD

      integer NITREJ
      real*8 darray(25)
      data darray /99.0d0,50.00d0,25.00d0,15.0d0, 10.00d0,
     .             8.00d0, 5.00d0, 3.50d0, 2.00d0, 1.50d0,
     .             1.20d0, 1.00d0, 0.80d0, 0.60d0, 0.45d0,
     .             0.35d0, 0.25d0, 0.20d0, 0.16d0, 0.12d0,
     .             0.09d0, 0.07d0, 0.05d0, 0.04d0, 0.03d0/

      real*8 mlim, mstep, qlim, qstep
      integer nlim, nstep, nliminit

      integer (kind=2), dimension(:,:), allocatable :: pk_map
      integer (kind=2), dimension(:,:), allocatable :: pk_mcp
      integer (kind=2), dimension(:,:), allocatable :: pk_cvg
      integer (kind=2), dimension(:,:), allocatable :: pk_cvu
      integer, dimension(:), allocatable :: pku

      integer PIXDIM, ix, iy, N, REJCOV, NPKx, NPKr, NPKs, NPKi
      integer NPK, ihx, ihy, ihxc, ihyc, NFND
      integer nhh
      real*8 xr, yr, xrmap, yrmap, xmin, xmax, ymin, ymax, duse, dmin
      real*8, dimension(:), allocatable :: xmap, ymap, pkxr, pkyr
      real*8 MINRAD
      integer hhist(100)

      integer   hash(_HASHDIM_,_HASHDIM_,_HASHDEP_)
      integer*2 nash(_HASHDIM_,_HASHDIM_)

      character*300 INMAT
      integer STAT
      logical LOC, AMP
      integer NLOC, NAMP
      logical FORCEMAT


      if (iargc().eq.0) then
         write(*,*)
         write(*,*)
         write(*,*)
         write(*,'('' xym2pm_Gaia.e GAIA_catalog HST_image.fits'')')
         write(*,*)
         write(*,'('' This program reads a Gaia catalog and '',
     .''a HST image, then it matches '')')
         write(*,'('' the HST list onto the Gaia-DR2 catalog.'')')
         write(*,*)
         write(*,*)
         write(*,*)
         write(*,'('' Optional parameters:'')')
         write(*,'('''')')
         write(*,'(''      "OBJECT=" -> (R.A.,Dec.) of the tangent '',
     .''point to project the Gaia-DR2'')')
         write(*,'(''                   catalog from a hard-coded '',
     .''list'')')
         write(*,*)
         write(*,'(''      "RACEN=" -> R.A. of the tangent point '',
     .''to project the Gaia-DR2 catalog'')')
         write(*,'(''                  (default chip center from '',
     .''the header)'')')
         write(*,*)
         write(*,'(''      "DECEN=" -> Dec. of the tangent point '',
     .''to project the Gaia-DR2 catalog'')')
         write(*,'(''                  (default chip center from '',
     .''the header)'')')
         write(*,*)
         write(*,'(''      "XCEN=" -> X position of the Gaia-DR2 '',
     .''tangent point (default 5000.0)'')')
         write(*,*)
         write(*,'(''      "YCEN=" -> Y position of the Gaia-DR2 '',
     .''tangent point (default 5000.0)'')')
         write(*,*)
         write(*,'(''      "PSCL=" -> Pixel scale to convert '',
     .''the Gaia-DR2 (R.A.,Dec.) into (X,Y)'')')
         write(*,'(''                 in pixel (if not given, '',
     .''it uses the pixel scale of the'')')
         write(*,'(''                 detector)'')')
         write(*,*)
         write(*,'(''      "SIZE=" -> Dimension of the '',
     .''master frame in pixel (default 10000.0)'')')
         write(*,*)
         write(*,'(''      "DISP=" -> Maximum displacement in pixel '',
     .''between the Gaia-DR2 positions '')')
         write(*,'(''                 and the globally-transformed '',
     .''positions (default 5.0)'')')
         write(*,*)
         write(*,'(''      "TIMEREF=" -> Reference epoch of the Gaia '',
     .''catalog (default 2016.0). '')')
         write(*,*)
         write(*,'(''      "TIME=" -> Reference epoch for the Gaia '',
     .''positions (default see TIMEREF). '')')
         write(*,'(''                 It uses the PMs to '',
     .''move the stars at this reference epoch.'')')
         write(*,'(''                 If "AUTO" is given, '',
     .''it uses the value from the image header'')')
         write(*,*)
         write(*,'(''      "MAT=" -> MAT file to help the initial '',
     .''match between Gaia and HST (default '')')
         write(*,'(''                none). It must be saved '',
     .''in the same directory defined by OUTPUT'')')
         write(*,'(''                (default '',
     .''the same of the image)'')')
         write(*,*)
         write(*,'(''      "USEMAT+" -> Use only stars in the MAT '',
     .''given in input with "MAT"'')')
         write(*,'(''                   (default FALSE)'')')
         write(*,*)
         write(*,'(''      "OUTDIR=" -> Output directory (default '',
     .''the same of the image)'')')
         write(*,*)
         write(*,'(''      "USESAT+" -> Use saturated stars in the '',
     .''HST image even after the initial'')')
         write(*,'(''                   match (default FALSE)'')')
         write(*,*)
         write(*,'(''      "NMIN=" -> Minimum number of stars '',
     .''needed to find a match (default 25)'')')
         write(*,*)
         write(*,'(''      "AMP+" -> Use local approaches: all '',
     .''stars in the same amplifier (default 25 stars)'')')
         write(*,*)
         write(*,'(''      "NAMP=" -> Number of closest stars '',
     .''in the same amplifier (default 25 stars)'')')
         write(*,*)
         write(*,'(''      "BRUTE=" -> Radius in arcsec within which '',
     .''to search for close-by objects in the '')')
         write(*,'(''                  Gaia catalog and find a match'',
     .''. It uses the header WCS and distortion '')')
         write(*,'(''                  correction to transform raw '',
     .''HST positions in (R.A.,Dec.) coordinates. '')')
         write(*,'(''                  Use only for sparse fields and'',
     .'' as last option'')')
         write(*,*)
         write(*,*)
         write(*,*)
         write(*,'('' Features:'')')
         write(*,*)
         write(*,'('' 1) It outputs a MAT file with the stars '',
     .''used for the transformations'')')
         write(*,'('' 2) If the MAT file for the given image is '',
     .''already present, it uses it at the beginning'')')
         write(*,'('' 3) You can give in input a MAT file of another '',
     .''image to help at the beginning'')')
         write(*,'('' 4) It outputs a LNK file with all stars '',
     .''found in common between Gaia and HST'')')
         write(*,'('' 5) The LNK file is ordered as the input Gaia '',
     .''file'')')
         write(*,*)
         write(*,*)
         write(*,*)
         stop
      endif

      OUTDIR = './'
      
      INMAT = 'NULL'
      
      PI_314 = 4.d0*atan(1.0d0)
      
      OBJECT = .false.
      OBJECTNAME = ''

      USESAT = .false.
      AMP = .false.
      NAMP = 25
      LOC = .false.
      NLOC = 25

      FORCEMAT = .false.

      nliminit = 25
      
      print*,''
      print*,''
      print*,''
      write(*,'(''  ----------------'')')         
      write(*,'('' |                |'')')         
      write(*,'('' | PROGRAM BEGINS |'')')
      write(*,'('' |                |'')')        
      write(*,'(''  ----------------'')')

      
      NARGs = iargc()

      call getarg(1,GAIAFILE)
     
      call getarg(2,IMGFILE)

! Retrieve some info from the image header
      call getinfo(IMGFILE,aux1,aux2,aux3,aux4,INSTRUME,DETECTOR,aux5)

      
      XY_XCEN = -50000.d0
      XY_YCEN = -50000.d0
      XY_PSCL =     0.0d0
      XY_RDAT =     0.0d0

      GAIA_RDAT = 2016.0d0

      RD_RCEN = -50000.d0
      RD_DCEN = -50000.d0

      PIXDIM = 0
      MINRAD = 0.d0
      RD_RAD = 0.d0

      
      do NARG = 3, iargc()
         STRING80 = ''
         call getarg(NARG,STRING80)
         if (index(STRING80,'XCEN=').gt.0) then
            read(STRING80(6:len_trim(STRING80)),*) XY_XCEN
            cycle
         endif
         if (index(STRING80,'YCEN=').gt.0) then
            read(STRING80(6:len_trim(STRING80)),*) XY_YCEN
            cycle
         endif
         if (index(STRING80,'PSCL=').gt.0) then
            read(STRING80(6:len_trim(STRING80)),*) XY_PSCL
            cycle
         endif
         if (index(STRING80,'SIZE=').gt.0) then
            read(STRING80(6:len_trim(STRING80)),*) PIXDIM
            cycle
         endif
         if (index(STRING80,'DISP=').gt.0) then
            read(STRING80(6:len_trim(STRING80)),*) MINRAD
            cycle
         endif
         if (index(STRING80,'MAT=').gt.0) then
            read(STRING80(5:len_trim(STRING80)),*) INMAT
            cycle
         endif
         if (index(STRING80,'USESAT+').gt.0) then
            USESAT = .true.
            cycle
         endif
         if (index(STRING80,'USESAT-').gt.0) then
            USESAT = .false.
            cycle
         endif
         if (index(STRING80,'AMP+').gt.0) then
            AMP = .true.
            cycle
         endif
         if (index(STRING80,'AMP-').gt.0) then
            AMP = .false.
            cycle
         endif
         if (index(STRING80,'LOC+').gt.0) then
            LOC = .true.
            cycle
         endif
         if (index(STRING80,'LOC-').gt.0) then
            LOC = .false.
            cycle
         endif
         if (index(STRING80,'USEMAT+').gt.0) then
            FORCEMAT = .true.
            cycle
         endif
         if (index(STRING80,'USEMAT-').gt.0) then
            FORCEMAT = .false.
            cycle
         endif
         if (index(STRING80,'NAMP=').gt.0) then
            read(STRING80(6:len_trim(STRING80)),*) NAMP
            cycle
         endif
         if (index(STRING80,'NLOC=').gt.0) then
            read(STRING80(6:len_trim(STRING80)),*) NLOC           
            cycle
         endif
         if (index(STRING80,'NMIN=').gt.0) then
            read(STRING80(6:len_trim(STRING80)),*) nliminit
            cycle
         endif
         if (index(STRING80,'BRUTE=').gt.0) then
            read(STRING80(7:len_trim(STRING80)),*) RD_RAD
            cycle
         endif
         if (index(STRING80,'OUTDIR=').gt.0) then
            OUTDIR = STRING80(8:len_trim(STRING80))
            if (OUTDIR(len_trim(OUTDIR):len_trim(OUTDIR)).ne.'/')
     .          OUTDIR(len_trim(OUTDIR)+1:len_trim(OUTDIR)+1) = '/'
            cycle
         endif
         if (index(STRING80,'TIMEREF=').gt.0) then
            read(STRING80(9:len_trim(STRING80)),*) GAIA_RDAT
            cycle
         endif
         if (index(STRING80,'TIME=').gt.0) then
            if (STRING80(6:10).eq.'AUTO') then
               if (aux5.gt.50) then
                  XY_RDAT = 1900.d0 + aux5
               else
                  XY_RDAT = 2000.d0 + aux5
               endif
            else
               read(STRING80(6:len_trim(STRING80)),*) XY_RDAT
            endif
            cycle
         endif
         if (index(STRING80,'RACEN=').gt.0) then
            RA_SEXIG = .false.            
            STRING_RA = STRING80(7:80)
            do i = 1, 20
               if (STRING_RA(i:i).eq.':') then
                  RA_SEXIG = .true.
                  STRING_RA(i:i) = ' '
               endif
            enddo
            if (RA_SEXIG) then
               read(STRING_RA,*) RAh, RAm, RAs
               RD_RCEN = 15.d0*(RAh + RAm/60.0d0 + RAs/3600.0d0)
            else
               read(STRING_RA,*) RD_RCEN
            endif
            cycle
         endif
         if (index(STRING80,'DECEN=').gt.0) then
            STRING_DE = STRING80(7:80)
            DE_SEXIG = .false.
            DE_NEG   = .false.
            do i = 1, 20
               if (STRING_DE(i:i).eq.'-') then
                  DE_NEG = .true.
                  STRING_DE(i:i) = ' '
               endif
               if (STRING_DE(i:i).eq.':') then
                  DE_SEXIG = .true.
                  STRING_DE(i:i) = ' '
               endif
            enddo
            if (DE_SEXIG) then
               read(STRING_DE,*) DEd, DEm, DEs
               RD_DCEN = (DEd + DEm/60.0d0 + DEs/3600.0d0)
            else
               read(STRING_DE,*) RD_DCEN
            endif
            if (DE_NEG) RD_DCEN = -RD_DCEN
            cycle
         endif
         if (index(STRING80,'OBJECT=').gt.0) then
            OBJECTNAME = STRING80(8:40)
            call object2radec(OBJECTNAME,STRING_RA,STRING_DE)
            RA_SEXIG = .false.
            do i = 1, 20
               if (STRING_RA(i:i).eq.':') then
                  RA_SEXIG = .true.
                  STRING_RA(i:i) = ' '
               endif
            enddo
            if (RA_SEXIG) then
               read(STRING_RA(8:40),*) RAh, RAm, RAs
               RD_RCEN = 15.d0*(RAh + RAm/60.0d0 + RAs/3600.0d0)
            else
               read(STRING_RA(8:40),*) RD_RCEN
            endif
            DE_SEXIG = .false.
            DE_NEG   = .false.
            do i = 1, 20
               if (STRING_DE(i:i).eq.'-') then
                  DE_NEG = .true.
                  STRING_DE(i:i) = ' '
               endif
               if (STRING_DE(i:i).eq.':') then
                  DE_SEXIG = .true.
                  STRING_DE(i:i) = ' '
               endif
            enddo
            if (DE_SEXIG) then
               read(STRING_DE(8:40),*) DEd, DEm, DEs
               RD_DCEN = (DEd + DEm/60.0d0 + DEs/3600.0d0)
            else
               read(STRING_DE(8:40),*) RD_DCEN
            endif
            if (DE_NEG) RD_DCEN = -RD_DCEN
            cycle
         endif
         print*,''
         write(*,'('' ARGUMENT -- '',a,'' -- NOT ACCEPTED! STOP!'')')
     .        STRING80(1:len_trim(STRING80))
         print*,''
         stop
      enddo

      
      print*,''
      write(*,'('' WORKING WITH IMAGE: '',a)')
     .     IMGFILE(1:len_trim(IMGFILE))
      print*,''
      write(*,'(''         INSTRUMENT: '',a)')
     .     INSTRUME(2:len_trim(INSTRUME)-1)
      write(*,'(''           DETECTOR: '',a)')
     .     DETECTOR(2:len_trim(DETECTOR)-1)

      
      if (RD_RAD.gt.0.d0) then
         XYMFILE = IMGFILE(1:(len_trim(IMGFILE)-4)) // 'XYmqxyrd'
         inquire(file=XYMFILE(1:(len_trim(XYMFILE))),exist=RES)
         if (.not.RES) then
            print*,''
            write(*,'(''    FILE '',a,'' NOT FOUND FOR BRUTE MODE'',
     .''! STOP!'')') XYMFILE(1:len_trim(XYMFILE))
            print*,''
            stop
         endif
      else
         XYMFILE = IMGFILE(1:(len_trim(IMGFILE)-4)) // 'XYmqxy'
         inquire(file=XYMFILE(1:(len_trim(XYMFILE))),exist=RES)
         if (.not.RES) then
            XYMFILE = IMGFILE(1:(len_trim(IMGFILE)-4)) // 'XYmqxyrd'
            inquire(file=XYMFILE(1:(len_trim(XYMFILE))),exist=RES)
            if (.not.RES) then
               print*,''
               write(*,'(''    FILES '',a,'' OR '',a,'' NOT FOUND!'',
     .'' STOP!'')') XYMFILE(1:len_trim(XYMFILE)-2),
     .              XYMFILE(1:len_trim(XYMFILE))
               print*,''
               stop
            endif
         endif
      endif

      
      print*,''
      write(*,'(''    READING XYMFILE: '',a)')
     .     XYMFILE(1:len_trim(XYMFILE))


      NSTARs = 0
      open(47,file=XYMFILE,status='old')
      do
         read(47,'(200a)',iostat=STAT) INSTRING
         if (IS_IOSTAT_END(STAT)) exit
         if (INSTRING(1:1).eq.'#') cycle
         NSTARs = NSTARs + 1
      enddo
      rewind(47)
      allocate(xchsto(NSTARs))
      allocate(ychsto(NSTARs))
      allocate(mhsto(NSTARs))
      allocate(qhsto(NSTARs))
      allocate(xrhsto(NSTARs))
      allocate(yrhsto(NSTARs))
      allocate(nhsto(NSTARs))
      allocate(rhsto(NSTARs))
      allocate(dhsto(NSTARs))
      xchsto = 0.d0
      ychsto = 0.d0
      mhsto = 0.d0
      qhsto = 0.d0
      xrhsto = 0.d0
      yrhsto = 0.d0
      rhsto = 0.d0
      dhsto = 0.d0
      nhsto = 0
      NSTARs = 0
      do
         read(47,'(200a)',iostat=STAT) INSTRING
         if (IS_IOSTAT_END(STAT)) exit
         if (INSTRING(1:1).eq.'#') cycle
         NSTARs = NSTARs + 1
! Read the info from the HST file
         if (RD_RAD.gt.0.d0) then
            read(INSTRING,*) xchsto(NSTARs), ychsto(NSTARs), 
     .                       mhsto(NSTARs), qhsto(NSTARs),
     .                       xrhsto(NSTARs), yrhsto(NSTARs),
     .                       rhsto(NSTARs), dhsto(NSTARs)
         else
            read(INSTRING,*) xchsto(NSTARs), ychsto(NSTARs), 
     .                       mhsto(NSTARs), qhsto(NSTARs),
     .                       xrhsto(NSTARs), yrhsto(NSTARs)
         endif
         nhsto(NSTARs) = NSTARs
      enddo
      close(47)

      print*,''
      write(*,'(''    SATURATED STARS: '',i5)')
     .     count(mhsto.lt.-9.and.qhsto.eq.0)
      write(*,'(''  UNSATURATED STARS: '',i5)')
     .     count(mhsto.lt.0.and.qhsto.gt.0)

      
! Radius that would cover the detector size (more or less...)
      RAD_DET = 0.0d0

! If not given in input, Gaia will be projected wrt a given point found in the header
      if (RD_RCEN.eq.-50000.d0) RD_RCEN = aux3
      if (RD_DCEN.eq.-50000.d0) RD_DCEN = aux4

! If not given in input, the tangent point is set at (5000,5000) pixel
      if (XY_XCEN.eq.-50000.d0) XY_XCEN = 5000.d0
      if (XY_YCEN.eq.-50000.d0) XY_YCEN = 5000.d0

! If not given in input, Gaia positions are left at epoch GAIA_RDAT
      if (XY_RDAT.eq.0.0) XY_RDAT = GAIA_RDAT

! If not given in input, the master frame will have size (10000,10000) pixel
      if (PIXDIM.eq.0) PIXDIM = 10000

! If not given in input, all stars with a maximum displacement of 5 pixels are kept in the LNK file
      if (MINRAD.eq.0d0) MINRAD = 5.d0
      
! Tangent point for Gaia
      RA_CEN = RD_RCEN
      DE_CEN = RD_DCEN
      
! Radius used to find the initial match. It is based on the size of the detector
      RAD_DET = 0.d0

      if (index(INSTRUME,'ACS').gt.0) then
         if (index(DETECTOR,'HRC').gt.0) then
            if (XY_PSCL.eq.0) XY_PSCL = 0.029d0
            RAD_DET = 0.01d0
         else
            if (index(DETECTOR,'WFC').gt.0) then
            if (XY_PSCL.eq.0) XY_PSCL = 0.050d0
               RAD_DET = 0.06d0
            else
               print*,'ACS DETECTOR NOT FOUND'
               stop
            endif
         endif
      else
         if (index(INSTRUME,'WFC3').gt.0) then
            if (index(DETECTOR,'UVIS').gt.0) then
               if (XY_PSCL.eq.0) XY_PSCL = 0.040d0
               RAD_DET = 0.045d0
            else
               if (index(DETECTOR,'IR').gt.0) then
                  if (XY_PSCL.eq.0) XY_PSCL = 0.120d0
                  RAD_DET = 0.035d0
               else
                  print*,'WFC3 DETECTOR NOT FOUND'
                  stop
               endif
            endif
         else
            print*,'CAMERA NOT SUPPORTED (YET)...'
         endif
      endif

      
! Preliminary position of the image based on the header -> RD and XY (I don't care of the PMs)
      RD_RCEN_LOC = aux3
      RD_DCEN_LOC = aux4
      dra = rd2x(RD_RCEN_LOC,RD_DCEN_LOC,RD_RCEN,RD_DCEN)
      dde = rd2y(RD_RCEN_LOC,RD_DCEN_LOC,RD_RCEN,RD_DCEN)
      XY_XCEN_LOC = XY_XCEN - dra*3600.d0/XY_PSCL
      XY_YCEN_LOC = XY_YCEN + dde*3600.d0/XY_PSCL
      
      print*,''
      print*,''
      print*,''
      write(*,'(''  -------------------'')')         
      write(*,'('' |                   |'')')         
      write(*,'('' | MASTER FRAME INFO |'')')
      write(*,'('' |                   |'')')        
      write(*,'(''  -------------------'')')
      print*,''
      write(*,'(''           X_CENTER: ''f8.2)') XY_XCEN
      write(*,'(''           Y_CENTER: ''f8.2)') XY_YCEN
      write(*,'(''          RA_CENTER: ''f13.8)') RA_CEN
      write(*,'(''         DEC_CENTER: ''f13.8)') DE_CEN
      write(*,'(''    SCALE (mas/pix): ''f8.2)') XY_PSCL*1000.d0

! Read the Gaia catalog
      Os = 0
      open(47,file=GAIAFILE,status='old')
      do
         read(47,'(200a)',iostat=STAT) INSTRING
         if (IS_IOSTAT_END(STAT)) exit
         if (INSTRING(1:1).eq.'#') cycle
         Os = Os + 1
      enddo
      rewind(47)
      allocate(  xpm_o(Os))
      allocate(  ypm_o(Os))
      allocate( expm_o(Os))
      allocate( eypm_o(Os))
      allocate(    x_o(Os))
      allocate(    y_o(Os))
      allocate(   ex_o(Os))
      allocate(   ey_o(Os))
      allocate(   ra_o(Os))
      allocate(   de_o(Os))
      allocate(  era_o(Os))
      allocate(  ede_o(Os))
      allocate(  dra_o(Os))
      allocate(  dde_o(Os))
      allocate(   mg_o(Os))
      allocate( mura_o(Os))
      allocate( mude_o(Os))
      allocate(emura_o(Os))
      allocate(emude_o(Os))
      allocate(ngaiao(Os))
      allocate(usegaia(Os))
      xpm_o = 0.d0
      ypm_o = 0.d0
      expm_o = 0.d0
      eypm_o = 0.d0
      x_o = 0.d0
      y_o = 0.d0
      ex_o = 0.d0
      ey_o = 0.d0
      ra_o = 0.d0
      de_o = 0.d0
      era_o = 0.d0
      ede_o = 0.d0
      dra_o = 0.d0
      dde_o = 0.d0
      mg_o = 0.d0
      mura_o = 0.d0
      mude_o = 0.d0
      emura_o = 0.d0
      emude_o = 0.d0
      ngaiao = 0
      usegaia = 1
      Os = 0
      do
         read(47,'(200a)',iostat=STAT) INSTRING
         if (IS_IOSTAT_END(STAT)) exit
         if (INSTRING(1:1).eq.'#') cycle
         Os = Os + 1
         read(INSTRING,*) ra_o(Os), era_o(Os), de_o(Os), ede_o(Os),
     .        mura_o(Os), emura_o(Os),
     .        mude_o(Os), emude_o(Os),
     .        mg_o(Os), usegaia(Os)
      enddo
      close(47)

! Compute the (X,Y) positions at epoch GAIA_RDAT and, if required, at the epoch XY_RDAT
      do Ou = 1, Os
! Project (R.A.,Dec.) onto the tangent plane centered in (RA_CEN,DE_CEN)        
         dra = rd2x(ra_o(Ou),de_o(Ou),RA_CEN,DE_CEN)
         dde = rd2y(ra_o(Ou),de_o(Ou),RA_CEN,DE_CEN)
         dra_o(Ou) = dra*3600.d0
         dde_o(Ou) = dde*3600.d0

         dYEARS = XY_RDAT - GAIA_RDAT

         XFRAME = 0.d0
         YFRAME = 0.d0
         XFRAME = XY_XCEN - dra*3600.d0/XY_PSCL
         YFRAME = XY_YCEN + dde*3600.d0/XY_PSCL
         x_o(Ou) = XFRAME
         y_o(Ou) = YFRAME
         ex_o(Ou) = era_o(Ou)/1000./XY_PSCL
         ey_o(Ou) = ede_o(Ou)/1000./XY_PSCL
         if (dYEARS.ne.0.d0) then
            if (mura_o(Ou).ne.0.d0.and.mude_o(Ou).ne.0.0d0.and.
     .          emura_o(Ou).ne.0.d0.and.emude_o(Ou).ne.0.0d0) then
               XFRAME = XY_XCEN - dra*3600/XY_PSCL
               YFRAME = XY_YCEN + dde*3600/XY_PSCL
               XFRAME = XFRAME + (-mura_o(Ou)/1000./XY_PSCL)*dYEARS
               YFRAME = YFRAME + ( mude_o(Ou)/1000./XY_PSCL)*dYEARS
! ML -> At the moment, I'm not including the contribution of the PM errors
               expm_o(Ou) = era_o(Ou)/1000./XY_PSCL
               eypm_o(Ou) = ede_o(Ou)/1000./XY_PSCL
            else
               XFRAME = 0.d0
               YFRAME = 0.d0
               expm_o(Ou) = 0.d0
               eypm_o(Ou) = 0.d0
            endif
         endif
         xpm_o(Ou) = XFRAME
         ypm_o(Ou) = YFRAME
         ngaiao(Ou) = Ou
      enddo


      
! First guess with saturated stars

      print*,''
      print*,''
      print*,''
      write(*,'(''  -----------'')')         
      write(*,'('' |           |'')')         
      write(*,'('' | GAIA INFO |'')')
      write(*,'('' |           |'')')        
      write(*,'(''  -----------'')')
      print*,''
      write(*,'(''    RA POSITION OF THE HST IMAGE:'',f13.8)')
     .     RD_RCEN_LOC
      write(*,'(''   DEC POSITION OF THE HST IMAGE:'',f13.8)')
     .     RD_DCEN_LOC
      
      xgaia = pack(xpm_o,xpm_o.gt.0.d0.and.ypm_o.gt.0.d0.and.
     .     sqrt((xpm_o-XY_XCEN_LOC)**2+(ypm_o-XY_YCEN_LOC)**2).lt.
     .     (RAD_DET*3600.d0/XY_PSCL+200.d0))
      ygaia = pack(ypm_o,xpm_o.gt.0.d0.and.ypm_o.gt.0.d0.and.
     .     sqrt((xpm_o-XY_XCEN_LOC)**2+(ypm_o-XY_YCEN_LOC)**2).lt.
     .     (RAD_DET*3600.d0/XY_PSCL+200.d0))
      mgaia = pack(mg_o,xpm_o.gt.0.d0.and.ypm_o.gt.0.d0.and.
     .     sqrt((xpm_o-XY_XCEN_LOC)**2+(ypm_o-XY_YCEN_LOC)**2).lt.
     .     (RAD_DET*3600.d0/XY_PSCL+200.d0))

      write(*,'(''                      GAIA EPOCH:'',f7.1)') GAIA_RDAT
      
      write(*,'(''                    N GAIA STARS:'',i8)') size(x_o)
      write(*,'(''  N GAIA STARS FOR INITIAL MATCH:'',i8)') size(xgaia)

      print*,''
      if (XY_RDAT.eq.GAIA_RDAT) then
         write(*,'(''            EPOCH GAIA POSITIONS: '',f7.2,
     .'' (ALL STARS USED)'')') XY_RDAT
      else
         write(*,'(''            EPOCH GAIA POSITIONS: '',f7.2,
     .'' (ONLY STARS WITH PM USED)'')') XY_RDAT
      endif
      

      NMATs  = 0
      NITREJ = 5

      
      allocate(x1c(_NMATMX_))
      allocate(y1c(_NMATMX_))
      allocate(m1c(_NMATMX_))
      allocate(x2c(_NMATMX_))
      allocate(y2c(_NMATMX_))
      allocate(m2c(_NMATMX_))
      allocate(dmc(_NMATMX_))
      x1c = 0.d0
      y1c = 0.d0
      m1c = 0.d0
      x2c = 0.d0
      y2c = 0.d0
      m2c = 0.d0

      MATFILE = OUTDIR(1:(len_trim(OUTDIR))) //
     .          IMGFILE(1:(len_trim(IMGFILE)-4)) // 'MAT'
      inquire(file=MATFILE(1:(len_trim(MATFILE))),exist=RES)
      if (RES.and.INMAT(1:4).eq.'NULL') then
         print*,''
         print*,''
         print*,''
         write(*,'(''    A MAT FILE ALREADY EXISTS FOR THIS IMAGE! '',
     .''USE IT AT THE BEGINNING...'')')
         NMATs = 0
         open(40,file=MATFILE,status='old')
         do
            read(40,'(200a)',iostat=STAT) INSTRING
            if (IS_IOSTAT_END(STAT)) exit
            if (INSTRING(1:1).eq.'#') cycle
            NMATs = NMATs + 1
            read(INSTRING,*) x1c(NMATs), y1c(NMATs),
     .                       x2c(NMATs), y2c(NMATs),
     .                       m1c(NMATs), m2c(NMATs)
         enddo
         close(40)         
      endif

      
      if (INMAT(1:4).ne.'NULL') then
         print*,''
         write(*,'(''    A MAT FILE HAS BEEN GIVEN IN '',
     .''INPUT FOR THIS IMAGE! USE IT AT THE BEGINNING...'')')
         NMATs = 0
         open(40,file=INMAT,status='old')
         do
            read(40,'(200a)',iostat=STAT) INSTRING
            if (IS_IOSTAT_END(STAT)) exit
            if (INSTRING(1:1).eq.'#') cycle
            NMATs = NMATs + 1
            read(INSTRING,*) x1c(NMATs), y1c(NMATs),
     .                       x2c(NMATs), y2c(NMATs),
     .                       m1c(NMATs), m2c(NMATs)
         enddo
         close(40)
         if (FORCEMAT) then
            print*,''
            write(*,'(''    "USEMAT" IS ON... '',
     .''SKIPPING INITIAL STAGES...'')')
            goto 2020
         endif
      endif
      
      
      print*,''
      print*,''
      print*,''

      write(*,'(''  ------------------------------------------'')')         
      write(*,'('' |                                          |'')')         
      write(*,'('' | SATURATED STARS INCLUDED IN THE HST LIST |'')')
      write(*,'('' |                                          |'')')        
      write(*,'(''  ------------------------------------------'')')    

      
! Use proximity in Equatorial coordinates -> Use it only in sparse field and as last option...
      if (RD_RAD.gt.0.d0) then

         write(*,'(''  -------------------------------------------'')')         
         write(*,'('' |                                           |'')')         
         write(*,'('' | PROXIMITY MATCH IN EQUATORIAL COORDINATES |'')')
         write(*,'('' |                                           |'')')        
         write(*,'(''  -------------------------------------------'')')

         if ((index(INSTRUME,'ACS').gt.0.and.
     .        index(DETECTOR,'WFC').gt.0).or.
     .       (index(INSTRUME,'WFC3').gt.0.and.
     .        index(DETECTOR,'UVIS').gt.0)) then
            
            print*,''
            write(*,'('' SEARCHING RADIUS: '',f5.3,'' arcsec'')') RD_RAD
            
            xhst = pack(rhsto,mhsto.le.-8.d0.and.
     .           qhsto.ge.0.00d0.and.qhsto.le.0.3)
            yhst = pack(dhsto,mhsto.le.-8.d0.and.
     .           qhsto.ge.0.00d0.and.qhsto.le.0.3)
            nhst = pack(nhsto,mhsto.le.-8.d0.and.
     .           qhsto.ge.0.00d0.and.qhsto.le.0.3)
            ragaia = pack(ra_o,xpm_o.gt.0.d0.and.ypm_o.gt.0.d0.and.
     .           sqrt((xpm_o-XY_XCEN_LOC)**2+(ypm_o-XY_YCEN_LOC)**2).lt.
     .           (RAD_DET*3600.d0/XY_PSCL+200.d0))
            degaia = pack(de_o,xpm_o.gt.0.d0.and.ypm_o.gt.0.d0.and.
     .           sqrt((xpm_o-XY_XCEN_LOC)**2+(ypm_o-XY_YCEN_LOC)**2).lt.
     .           (RAD_DET*3600.d0/XY_PSCL+200.d0))
            ngaia = pack(ngaiao,xpm_o.gt.0.d0.and.ypm_o.gt.0.d0.and.
     .           sqrt((xpm_o-XY_XCEN_LOC)**2+(ypm_o-XY_YCEN_LOC)**2).lt.
     .           (RAD_DET*3600.d0/XY_PSCL+200.d0))

            NREFs  = size(ragaia)
            NSTARs = size(xhst)

            allocate(n1match(_NSTMAX_),n2match(_NSTMAX_))
            n1match = 0
            n2match = 0

            call jedimatch(xhst,yhst,nhst,NSTARs,
     .                     ragaia,degaia,ngaia,NREFs,
     .                     n1match,n2match,RD_RAD,RA_CEN,DE_CEN)

            if (count(n1match.gt.0.and.n2match.gt.0).gt.0) then
               NMATs = 0
               do i = 1, count(n1match.gt.0.and.n2match.gt.0)
                  NMATs = NMATs+1
                  x1c(NMATs) = xpm_o(n2match(i))
                  y1c(NMATs) = ypm_o(n2match(i))
                  m1c(NMATs) =  mg_o(n2match(i))
                  x2c(NMATs) = xchsto(n1match(i))
                  y2c(NMATs) = ychsto(n1match(i))
                  m2c(NMATs) =  mhsto(n1match(i))
               enddo
            endif

            if (count(n1match.gt.0.and.n2match.gt.0).gt.0) then
               print*,''
               write(*,'('' SUCCESS! '',i3,'' STARS FOUND'')') NMATs
            else
               print*,''
               write(*,'('' FAILED! NO STARS FOUND'')')
            endif
            
            deallocate(xhst,yhst,nhst,ragaia,degaia,ngaia)
            deallocate(n1match,n2match)

         else
            print*,''
            write(*,'('' SKIP! NOT IMPLEMENTED FOR THIS '',
     .''CAMERA/INSTRUMENT'')')
         endif
            
      endif

     
      xhst = pack(xchsto,mhsto.le.-8.d0.and.
     .            qhsto.ge.0.00d0.and.qhsto.le.0.3)
      yhst = pack(ychsto,mhsto.le.-8.d0.and.
     .            qhsto.ge.0.00d0.and.qhsto.le.0.3)
      mhst = pack(mhsto,mhsto.le.-8.d0.and.
     .            qhsto.ge.0.00d0.and.qhsto.le.0.3)

      
      NREFs  = size(xgaia)
      NSTARs = size(xhst)

      
! This routine finds the match between two lists of stars using different techniques
      call miracle_match(
     .        xgaia,ygaia,mgaia,NREFs,
     .        xhst,yhst,mhst,NSTARs,
     .        x1c,y1c,m1c,x2c,y2c,m2c,NMATs,NITREJ,1)

! This routine finds the coefficients of the six-parameter linear transformations between the two lists
      call glob_fit6nrDP(x2c,y2c,x1c,y1c,NMATs,
     .                   AG, BG, CG, DG, Xo, Yo, Wo, Zo)

      dmc = m1c-m2c
      call barsig(dmc,NMATs,bar,sig,NMATu,2.0d0)

      DELTA = AG*DG-BG*CG

      print*,''
  
      if (NMATs.eq.0) then
         write(*,'('' NO MATCH FOUND! EXIT!'')')
         print*,''
         stop
      endif

      write(*,'('' HOORAY! MATCH FOUND!'')')
      
      print*,''
      write(*,'(''            STARS FOUND IN COMMON: '',i6)') NMATs
      write(*,'(''            TRANSFORMATION MATRIX: X_2 = '',
     .''AG*(X_1-Xo)+BG*(Y_1-Yo)+Wo'')')
      write(*,'(''                                   Y_2 = '',
     .''CG*(X_1-Xo)+DG*(Y_1-Yo)+Zo'')')
      print*,''
      write(*,'(''                               AG: ''f14.11)') AG
      write(*,'(''                               BG: ''f14.11)') BG
      write(*,'(''                               CG: ''f14.11)') CG
      write(*,'(''                               DG: ''f14.11)') DG
      write(*,'(''                               Xo: ''f14.4)') Xo
      write(*,'(''                               Yo: ''f14.4)') Yo
      write(*,'(''                               Wo: ''f14.4)') Wo
      write(*,'(''                               Zo: ''f14.4)') Zo
      print*,''
      write(*,'(''                   ROTATION (deg):'',f7.2)')
     .               atan2dj((BG-CG),(AG+DG))
      write(*,'(''                PIXEL-SCALE RATIO:'',f12.8)')
     .     DELTA/abs(DELTA)*sqrt(abs(DELTA))
      write(*,'(''   REAL IMG PIXEL SCALE (mas/pix):'',f12.8)')
     .     XY_PSCL*(DELTA/abs(DELTA)*sqrt(abs(DELTA)))*1000.d0
      print*,''
      write(*,'(''                     MAGNITUDE ZP:'',f7.2)')
     .     bar
      print*,''

      deallocate(xgaia,ygaia,mgaia,xhst,yhst,mhst)


      allocate(x1co(_NMATMX_))
      allocate(y1co(_NMATMX_))
      allocate(m1co(_NMATMX_))
      allocate(x2co(_NMATMX_))
      allocate(y2co(_NMATMX_))
      allocate(m2co(_NMATMX_))
      x1co = x1c
      y1co = y1c
      m1co = m1c
      x2co = x2c
      y2co = y2c
      m2co = m2c
      NMATso = NMATs

      
      print*,''
      print*,''

      
      xgaia = pack(xpm_o,xpm_o.gt.0.d0.and.ypm_o.gt.0.d0.and.
     .             usegaia.eq.1)
      ygaia = pack(ypm_o,xpm_o.gt.0.d0.and.ypm_o.gt.0.d0.and.
     .             usegaia.eq.1)
      mgaia = pack(mg_o,xpm_o.gt.0.d0.and.ypm_o.gt.0.d0.and.
     .             usegaia.eq.1)

      NREFs  = size(xgaia)
      NITREJ = 5
      
      if (.not.USESAT) then
         write(*,'(''  -------------------------------------------'')')         
         write(*,'('' |                                           |'')')         
         write(*,'('' | SATURATED STARS REMOVED FROM THE HST LIST |'')')
         write(*,'('' |                                           |'')')        
         write(*,'(''  -------------------------------------------'')')         
      
! I try a first test: if it doesn't work with unsaturated stars, too bad...
 
         xhst = pack(xchsto,mhsto.le.-8.d0.and.
     .               qhsto.gt.0.00d0.and.qhsto.le.0.3d0)
         yhst = pack(ychsto,mhsto.le.-8.d0.and.
     .               qhsto.gt.0.00d0.and.qhsto.le.0.3d0)
         mhst = pack(mhsto,mhsto.le.-8.d0.and.
     .               qhsto.gt.0.00d0.and.qhsto.le.0.3d0)
      
         NSTARs = size(xhst)

         call miracle_match(
     .        xgaia,ygaia,mgaia,NREFs,
     .        xhst,yhst,mhst,NSTARs,
     .        x1c,y1c,m1c,x2c,y2c,m2c,NMATs,NITREJ,0)

         call glob_fit6nrDP(x2c,y2c,x1c,y1c,NMATs,
     .        AG, BG, CG, DG, Xo, Yo, Wo, Zo)
         
         dmc = m1c-m2c
         call barsig(dmc,NMATs,bar,sig,NMATu,2.0d0)

         print*,''

         if (NMATs.eq.0) then
            write(*,'('' MATCH AVAILABLE ONLY WITH SATURATED '',
     .''STARS... NICE TRY...'')')
            print*,''
            stop
         endif

         deallocate(xhst,yhst,mhst)
      endif
      

! Iter begins

      mlim = -12.d0
      qlim = 0.05d0
      nlim = 500
      
      mstep = 0.5d0
      qstep = 0.05d0
      nstep = 50

      mlim = mlim-mstep

! I try to compute the best match possible. If it doesn't work, I relax the values,
! but down to reasonable thresholds...
      
 911  continue

      x1c = x1co
      y1c = y1co
      m1c = m1co
      x2c = x2co
      y2c = y2co
      m2c = m2co
      NMATs = NMATso
      
      write(*,'('' TESTING THE FOLLOWING LIMITS FOR HST:''
     .'' (MAG<'',f5.1,'' , QFIT<'',f4.2,'')'',a1)', advance='NO')
     .     mlim, qlim, achar(13)

      if (USESAT) then
         xhst = pack(xchsto,mhsto.le.mlim.and.
     .               qhsto.ge.0.00d0.and.qhsto.le.qlim)
         yhst = pack(ychsto,mhsto.le.mlim.and.
     .               qhsto.ge.0.00d0.and.qhsto.le.qlim)
         mhst = pack(mhsto,mhsto.le.mlim.and.
     .               qhsto.ge.0.00d0.and.qhsto.le.qlim)
      else
         xhst = pack(xchsto,mhsto.le.mlim.and.
     .               qhsto.gt.0.00d0.and.qhsto.le.qlim)
         yhst = pack(ychsto,mhsto.le.mlim.and.
     .               qhsto.gt.0.00d0.and.qhsto.le.qlim)
         mhst = pack(mhsto,mhsto.le.mlim.and.
     .               qhsto.gt.0.00d0.and.qhsto.le.qlim)
      endif
      
      NREFs  = size(xgaia)
      NSTARs = size(xhst)
      NITREJ = 5

      call miracle_match(
     .        xgaia,ygaia,mgaia,NREFs,
     .        xhst,yhst,mhst,NSTARs,
     .        x1c,y1c,m1c,x2c,y2c,m2c,NMATs,NITREJ,0)

      deallocate(xhst,yhst,mhst)

      if (NMATs.lt.nlim.and.nlim.ge.nliminit) then
! If the mag limit is not reached, try do decrease the mag limit and keep the rest as it is
         if (mlim.lt.21.5-bar) then
            mlim = mlim + mstep
            goto 911
         endif
! If the mag limit is reached, restart from the brightest stars and change the QFIT
         mlim = -12.5d0
         if (qlim.lt.0.3) then
            qlim = qlim + qstep
            goto 911
         endif
! If both the mag and the QFIT limits are reached, try the last option: look for less stars
         qlim = 0.05d0
         if (nlim.ge.nliminit) then
            if (nlim.lt.100) nstep = 1
            nlim = nlim - nstep
            goto 911
         endif
      endif

      deallocate(xgaia,ygaia,mgaia)
      deallocate(x1co,y1co,m1co,x2co,y2co,m2co)
      
      if (NMATs.lt.nliminit) then
         write(*,'('' OK, I TRIED... NO DECENT TRANSFORMATIONS ''
     .''WITHOUT SATURATED STARS'')')
         stop
      endif

      write(*,'('' HOORAY! BETTER MATCH FOUND!'',
     .'' WITH THE FOLLOWING LIMITS FOR HST:'',
     .'' (MAG<'',f5.1,'' , QFIT<'',f4.2,'')'')')
     .     mlim, qlim

      
      xgaia = pack(xpm_o,xpm_o.gt.0.d0.and.ypm_o.gt.0.d0.and.
     .     usegaia.eq.1)
      ygaia = pack(ypm_o,xpm_o.gt.0.d0.and.ypm_o.gt.0.d0.and.
     .     usegaia.eq.1)
      mgaia = pack(mg_o,xpm_o.gt.0.d0.and.ypm_o.gt.0.d0.and.
     .     usegaia.eq.1)

      xhst = pack(xchsto,mhsto.le.-8.d0.and.
     .            qhsto.ge.0.00d0.and.qhsto.le.0.3)
      yhst = pack(ychsto,mhsto.le.-8.d0.and.
     .            qhsto.ge.0.00d0.and.qhsto.le.0.3)
      mhst = pack(mhsto,mhsto.le.-8.d0.and.
     .            qhsto.ge.0.00d0.and.qhsto.le.0.3)

      NREFs  = size(xgaia)
      NSTARs = size(xhst)
      NITREJ = 5

      call miracle_match(
     .        xgaia,ygaia,mgaia,NREFs,
     .        xhst,yhst,mhst,NSTARs,
     .        x1c,y1c,m1c,x2c,y2c,m2c,NMATs,NITREJ,0)
            
      call glob_fit6nrDP(x2c,y2c,x1c,y1c,NMATs,
     .                   AG, BG, CG, DG, Xo, Yo, Wo, Zo)      

      deallocate(xgaia,ygaia,mgaia,xhst,yhst,mhst)
     
      xgaia = pack(xpm_o,xpm_o.gt.0.d0.and.ypm_o.gt.0.d0.and.
     .     usegaia.eq.1)
      ygaia = pack(ypm_o,xpm_o.gt.0.d0.and.ypm_o.gt.0.d0.and.
     .     usegaia.eq.1)
      mgaia = pack(mg_o,xpm_o.gt.0.d0.and.ypm_o.gt.0.d0.and.
     .     usegaia.eq.1)

      if (USESAT) then
         xhst = pack(xchsto,mhsto.le.mlim.and.
     .               qhsto.ge.0.00d0.and.qhsto.le.qlim)
         yhst = pack(ychsto,mhsto.le.mlim.and.
     .               qhsto.ge.0.00d0.and.qhsto.le.qlim)
         mhst = pack(mhsto,mhsto.le.mlim.and.
     .               qhsto.ge.0.00d0.and.qhsto.le.qlim)
      else
         xhst = pack(xchsto,mhsto.le.mlim.and.
     .               qhsto.gt.0.00d0.and.qhsto.le.qlim)
         yhst = pack(ychsto,mhsto.le.mlim.and.
     .               qhsto.gt.0.00d0.and.qhsto.le.qlim)
         mhst = pack(mhsto,mhsto.le.mlim.and.
     .               qhsto.gt.0.00d0.and.qhsto.le.qlim)
      endif
      
      NREFs  = size(xgaia)
      NSTARs = size(xhst)
      NITREJ = 5

      call miracle_match(
     .        xgaia,ygaia,mgaia,NREFs,
     .        xhst,yhst,mhst,NSTARs,
     .        x1c,y1c,m1c,x2c,y2c,m2c,NMATs,NITREJ,1)
            
      call glob_fit6nrDP(x2c,y2c,x1c,y1c,NMATs,
     .                   AG, BG, CG, DG, Xo, Yo, Wo, Zo)
      
      dmc = m1c-m2c
      call barsig(dmc,NMATs,bar,sig,NMATu,2.0d0)

      DELTA = AG*DG-BG*CG

      print*,''
      write(*,'(''            STARS FOUND IN COMMON: '',i6)') NMATs
      write(*,'(''            TRANSFORMATION MATRIX: X_2 = '',
     .''AG*(X_1-Xo)+BG*(Y_1-Yo)+Wo'')')
      write(*,'(''                                   Y_2 = '',
     .''CG*(X_1-Xo)+DG*(Y_1-Yo)+Zo'')')
      print*,''
      write(*,'(''                               AG: ''f14.11)') AG
      write(*,'(''                               BG: ''f14.11)') BG
      write(*,'(''                               CG: ''f14.11)') CG
      write(*,'(''                               DG: ''f14.11)') DG
      write(*,'(''                               Xo: ''f14.4)') Xo
      write(*,'(''                               Yo: ''f14.4)') Yo
      write(*,'(''                               Wo: ''f14.4)') Wo
      write(*,'(''                               Zo: ''f14.4)') Zo
      print*,''
      write(*,'(''                   ROTATION (deg):'',f7.2)')
     .               atan2dj((BG-CG),(AG+DG))
      write(*,'(''                PIXEL-SCALE RATIO:'',f12.8)')
     .     DELTA/abs(DELTA)*sqrt(abs(DELTA))
      write(*,'(''   REAL IMG PIXEL SCALE (mas/pix):'',f12.8)')
     .     XY_PSCL*(DELTA/abs(DELTA)*sqrt(abs(DELTA)))*1000.d0
      print*,''
      write(*,'(''                     MAGNITUDE ZP:'',f7.2)')
     .     bar
      print*,''


     
! For stars in common, I transform all HST positions onto the Gaia frame and I compute
! the positional residuals
      
      allocate(dr(NMATs))
      dr = 0.d0

      do NMAT = 1, NMATs
         xglob1 = AG*(x2c(NMAT)-Xo)+BG*(y2c(NMAT)-Yo)+Wo
         yglob1 = CG*(x2c(NMAT)-Xo)+DG*(y2c(NMAT)-Yo)+Zo
         dr(NMAT) = sqrt((x1c(NMAT)-xglob1)**2+
     .                   (y1c(NMAT)-yglob1)**2)
      enddo

! Mean and sigma of the residuals
      call barsig(dr,NMATs,bar,sig,NMATu,3.5d0)

      NITREJ = minloc(abs(darray-3.5d0*bar),1) + 1

      print*,''
      print*,''

      write(*,'(''  -----------------'')')         
      write(*,'('' |                 |'')')         
      write(*,'('' | SOME STATISTICS |'')')
      write(*,'('' |                 |'')')        
      write(*,'(''  -----------------'')')

      print*,''
      write(*,'('' AVERAGE POSITIONAL RESIDUALS: '',f6.3,'' PIXELS'')')
     .     bar
      print*,''
      write(*,'('' REMOVING STARS WITH POSITIONAL RESIDUALS '',
     .'' LARGER THAN '',f6.3,'' PIXELS (3.5 x AVERAGE)'')')
     .     darray(NITREJ)
      print*,''

      allocate(x1co(_NMATMX_))
      allocate(y1co(_NMATMX_))
      allocate(m1co(_NMATMX_))
      allocate(x2co(_NMATMX_))
      allocate(y2co(_NMATMX_))
      allocate(m2co(_NMATMX_))
      x1co = x1c
      y1co = y1c
      m1co = m1c
      x2co = x2c
      y2co = y2c
      m2co = m2c
      x1c = 0.d0
      y1c = 0.d0
      m1c = 0.d0
      x2c = 0.d0
      y2c = 0.d0
      m2c = 0.d0

! I remove all stars with residuals >3 times the average positional residual
      NMATu = 0
      do NMAT = 1, NMATs
         if (dr(NMAT).lt.darray(NITREJ)) then
            NMATu = NMATu + 1
            x1c(NMATu) = x1co(NMAT)
            y1c(NMATu) = y1co(NMAT)
            m1c(NMATu) = m1co(NMAT)
            x2c(NMATu) = x2co(NMAT)
            y2c(NMATu) = y2co(NMAT)
            m2c(NMATu) = m2co(NMAT)
         endif
      enddo
      NMATs = NMATu

      deallocate(x1co,y1co,m1co,x2co,y2co,m2co)

      if (NMATs.eq.0) then
         print*,''
         write(*,'('' OK, AFTER CLIPPING NO STARS ARE LEFT... ''
     .'' WERE THEY MISMATCHES?'')')
         stop
      endif
      
      print*,''
      print*,''

      write(*,'(''  --------------------------------'')')         
      write(*,'('' |                                |'')')         
      write(*,'('' | NEW TRANSFORMATIONS AFTER CLIP |'')')
      write(*,'('' |                                |'')')        
      write(*,'(''  --------------------------------'')')         

      deallocate(xgaia,ygaia,mgaia,xhst,yhst,mhst)
      
      xgaia = pack(xpm_o,xpm_o.gt.0.d0.and.ypm_o.gt.0.d0.and.
     .     usegaia.eq.1)
      ygaia = pack(ypm_o,xpm_o.gt.0.d0.and.ypm_o.gt.0.d0.and.
     .     usegaia.eq.1)
      mgaia = pack(mg_o,xpm_o.gt.0.d0.and.ypm_o.gt.0.d0.and.
     .     usegaia.eq.1)

      if (USESAT) then
         xhst = pack(xchsto,mhsto.le.mlim.and.
     .               qhsto.ge.0.00d0.and.qhsto.le.qlim)
         yhst = pack(ychsto,mhsto.le.mlim.and.
     .               qhsto.ge.0.00d0.and.qhsto.le.qlim)
         mhst = pack(mhsto,mhsto.le.mlim.and.
     .               qhsto.ge.0.00d0.and.qhsto.le.qlim)
      else
         xhst = pack(xchsto,mhsto.le.mlim.and.
     .               qhsto.gt.0.00d0.and.qhsto.le.qlim)
         yhst = pack(ychsto,mhsto.le.mlim.and.
     .               qhsto.gt.0.00d0.and.qhsto.le.qlim)
         mhst = pack(mhsto,mhsto.le.mlim.and.
     .               qhsto.gt.0.00d0.and.qhsto.le.qlim)
      endif
      
      NREFs  = size(xgaia)
      NSTARs = size(xhst)

      call miracle_match(
     .        xgaia,ygaia,mgaia,NREFs,
     .        xhst,yhst,mhst,NSTARs,
     .        x1c,y1c,m1c,x2c,y2c,m2c,NMATs,NITREJ,1)

      
      if (NMATs.eq.0) then
         print*,''
         write(*,'('' OK, AFTER CLIPPING NO STARS ARE LEFT... ''
     .'' WERE THEY MISMATCHES?'')')
         stop
      endif


      deallocate(xgaia,ygaia,mgaia,xhst,yhst,mhst)

! The program jumps here if a MAT file is given in input and it is forced to use it as it is
 2020 continue

      call glob_fit6nrDP(x2c,y2c,x1c,y1c,NMATs,
     .                   AG, BG, CG, DG, Xo, Yo, Wo, Zo)
      
      dmc = m1c-m2c
      call barsig(dmc,NMATs,bar,sig,NMATu,2.00d0)

      DELTA = AG*DG-BG*CG

      print*,''
      write(*,'(''            STARS FOUND IN COMMON: '',i6)') NMATs
      write(*,'(''            TRANSFORMATION MATRIX: X_2 = '',
     .''AG*(X_1-Xo)+BG*(Y_1-Yo)+Wo'')')
      write(*,'(''                                   Y_2 = '',
     .''CG*(X_1-Xo)+DG*(Y_1-Yo)+Zo'')')
      print*,''
      write(*,'(''                               AG: ''f14.11)') AG
      write(*,'(''                               BG: ''f14.11)') BG
      write(*,'(''                               CG: ''f14.11)') CG
      write(*,'(''                               DG: ''f14.11)') DG
      write(*,'(''                               Xo: ''f14.4)') Xo
      write(*,'(''                               Yo: ''f14.4)') Yo
      write(*,'(''                               Wo: ''f14.4)') Wo
      write(*,'(''                               Zo: ''f14.4)') Zo
      print*,''
      write(*,'(''                   ROTATION (deg):'',f7.2)')
     .               atan2dj((BG-CG),(AG+DG))
      write(*,'(''                PIXEL-SCALE RATIO:'',f12.8)')
     .     DELTA/abs(DELTA)*sqrt(abs(DELTA))
      write(*,'(''   REAL IMG PIXEL SCALE (mas/pix):'',f12.8)')
     .     XY_PSCL*(DELTA/abs(DELTA)*sqrt(abs(DELTA)))*1000.d0
      print*,''
      write(*,'(''                     MAGNITUDE ZP:'',f7.2)')
     .     bar
      print*,''


! Save MAT file
      
      print*,''
      print*,''
      write(*,'('' -----------------'')')
      write(*,'(''|                 |'')')
      write(*,'(''| SAVING MAT FILE |'')')
      write(*,'(''|                 |'')')
      write(*,'('' -----------------'')')
   
      MATFILE = OUTDIR(1:(len_trim(OUTDIR))) //
     .          IMGFILE(1:(len_trim(IMGFILE)-4)) // 'MAT'

      print*,''
      write(*,'('' SAVING: '',a)') MATFILE(1:(len_trim(MATFILE)))
      print*,''
      
      open(50,file=MATFILE,status='unknown')
      do NMAT = 1, NMATs
         xglob1 = AG*(x2c(NMAT)-Xo)+BG*(y2c(NMAT)-Yo)+Wo
         yglob1 = CG*(x2c(NMAT)-Xo)+DG*(y2c(NMAT)-Yo)+Zo
         write(50,922)
     .        x1c(NMAT), y1c(NMAT),                       ! Gaia positions
     .        x2c(NMAT), y2c(NMAT),                       ! HST distortion-corrected positions
     .        m1c(NMAT), m2c(NMAT),                       ! Gaia and HST magnitudes
     .        (x1c(NMAT)-xglob1), (y1c(NMAT)-yglob1),     ! Difference between the Gaia position and the HST positions transformed onto the Gaia reference frame
     .        xglob1, yglob1                              ! HST positions transformed onto the Gaia reference frame
      enddo
      close(50)
 922  format(10(f12.4,1x))
      
      
      print*,''
      print*,''
      print*,''
      write(*,'('' -----------------------------'')')
      write(*,'(''|                             |'')')
      write(*,'(''| FINDING ALL STARS IN COMMON |'')')
      write(*,'(''|                             |'')')
      write(*,'('' -----------------------------'')')

      allocate(pk_cvu(PIXDIM,PIXDIM))
      allocate(pk_cvg(PIXDIM,PIXDIM))
      allocate(pk_map(PIXDIM,PIXDIM))
      pk_cvu = 0
      pk_cvg = 0
      pk_map = 0

      if (index(INSTRUME,'ACS').gt.0.and.index(DETECTOR,'HRC').gt.0)
     .     call cvg_add_c(pk_cvu,7,PIXDIM,AG,BG,CG,DG,Xo,Yo,Wo,Zo)
      if (index(INSTRUME,'ACS').gt.0.and.index(DETECTOR,'ACS').gt.0)
     .     call cvg_add_c(pk_cvu,5,PIXDIM,AG,BG,CG,DG,Xo,Yo,Wo,Zo)
      if (index(INSTRUME,'WFC3').gt.0.and.index(DETECTOR,'UVIS').gt.0)
     .     call cvg_add_c(pk_cvu,8,PIXDIM,AG,BG,CG,DG,Xo,Yo,Wo,Zo)
      if (index(INSTRUME,'WFC3').gt.0.and.index(DETECTOR,'IR').gt.0)
     .     call cvg_add_c(pk_cvu,9,PIXDIM,AG,BG,CG,DG,Xo,Yo,Wo,Zo)
     
      NPKs = 0
      NPKx = 0
      NPKr = 0
      NPKi = 0
      
      N = 0
      do iy = 1, PIXDIM
         do ix = 1, PIXDIM
            pk_cvg(ix,iy) = pk_cvg(ix,iy)+pk_cvu(ix,iy)
            N = N + pk_cvu(ix,iy)
         enddo
      enddo
!      print*,'---> total covered pixels: ',N
!      print*,'---> total         NSTARs: ',NSTARs
      
      print*,' '
      do N = 1, size(xchsto)
         xr = AG*(xchsto(N)-Xo)+BG*(ychsto(N)-Yo) + Wo
         yr = CG*(xchsto(N)-Xo)+DG*(ychsto(N)-Yo) + Zo
         xrmap = xr
         yrmap = yr
         ix = nint(xrmap)
         iy = nint(yrmap)
         if (ix.ge.1.and.ix.le.PIXDIM.and.
     .       iy.ge.1.and.iy.le.PIXDIM.and.
     .       pk_cvu(ix,iy).eq.001) then
            if (pk_cvu(ix,iy).eq.1) then
               pk_map(ix,iy) = pk_map(ix,iy)+1
               NPKx   = NPKx + 1
            else
               REJCOV = REJCOV + 1
               NPKr   = NPKr + 1
            endif
         endif
      enddo
 
!      print*,' '
!      print*,'---> NSTARs: ',size(xchsto)
!      print*,'       NPKx: ',NPKx
!      print*,'       NPKr: ',NPKr
!      print*,'     REJCOV: ',REJCOV
!      print*,' '
      
      xgaia = pack(xpm_o,xpm_o.gt.0.d0.and.ypm_o.gt.0.d0)
      ygaia = pack(ypm_o,xpm_o.gt.0.d0.and.ypm_o.gt.0.d0)
      mgaia = pack(mg_o,xpm_o.gt.0.d0.and.ypm_o.gt.0.d0)
      ngaia = pack(ngaiao,xpm_o.gt.0.d0.and.ypm_o.gt.0.d0)
      NPKs = size(xgaia)

      allocate(pku(NPKs))
      pku = 0
      
      do NPK = 1, NPKs
         ix = nint(xgaia(NPK))
         iy = nint(ygaia(NPK))
         if (ix.ge.1.and.ix.le.PIXDIM.and.iy.ge.1.and.iy.le.PIXDIM)
     .        pku(NPK) = 1
      enddo

      allocate(xmap(size(xchsto)))
      allocate(ymap(size(ychsto)))
      xmap = 0.d0
      ymap = 0.d0
      
      do N = 1, size(xchsto)
         xr = AG*(xchsto(N)-Xo)+BG*(ychsto(N)-Yo)+Wo
         yr = CG*(xchsto(N)-Xo)+DG*(ychsto(N)-Yo)+Zo
         xmap(N) = xr
         ymap(N) = yr
      enddo
      
!      write(*,36) NIMi,NSTARs,NTRY,FILE(1:25)
!      write(*,37) '   Xh: ',minarr(xloc,NSTARs), maxarr(xloc,NSTARs)
!      write(*,37) '   Yh: ',minarr(yloc,NSTARs), maxarr(yloc,NSTARs)
!      write(*,37) '   Zh: ',minarr(zloc,NSTARs), maxarr(zloc,NSTARs)
!      write(*,* ) '   Fh: ',CHIPNO,IFILT
      xmin = minval(xchsto,NSTARs)
      ymin = minval(ychsto,NSTARs)
      xmax = maxval(xchsto,NSTARs)
      ymax = maxval(ychsto,NSTARs)
      if (xmax.lt.xmin+_HASHDIM_) xmax = xmin+_HASHDIM_
      if (ymax.lt.ymin+_HASHDIM_) ymax = ymin+_HASHDIM_

!      print*,'THROW THE STARS INTO THE HASH TABLE...  NSTARs: ',NSTARs
      nash   = 0
 
      print*,''
      print*,''
      write(*,'('' -----------------'')')
      write(*,'(''|                 |'')')
      write(*,'(''| SAVING LNK FILE |'')')
      write(*,'(''|                 |'')')
      write(*,'('' -----------------'')')

!      print*,'xmin/max: ',xmin,xmax
!      print*,'ymin/max: ',ymin,ymax
      do N = 1, size(xchsto)
         ihx = 1 + (_HASHDIM_-1)*(xchsto(N)-xmin)/(xmax-xmin)
         ihy = 1 + (_HASHDIM_-1)*(ychsto(N)-ymin)/(ymax-ymin)
         if (ihx.lt.1) ihx = 1
         if (ihy.lt.1) ihy = 1
         if (ihx.gt._HASHDIM_) ihx = _HASHDIM_
         if (ihy.gt._HASHDIM_) ihy = _HASHDIM_
         nash(ihx,ihy) = nash(ihx,ihy) + 1
         if (nash(ihx,ihy).gt._HASHDEP_) nash(ihx,ihy) = _HASHDEP_
         if (nash(ihx,ihy).gt._HASHDEP_.and.
     .       ihx.gt.1.and.ihy.gt.1) then
            print*,'           '
            print*,'_HASHDEP_: ',_HASHDEP_
            NSTAR = hash(ihx,ihy,nash(ihx,ihy))
            print*,'    NSTAR: ',N, xchsto(N) ,
     .                              ychsto(N) ,mhsto(N)
            print*,'     NASH: ',NSTAR,xchsto(NSTAR),
     .                                 ychsto(NSTAR),mhsto(NSTAR)
            print*,'      ihx: ',ihx,xmin,xmax
            print*,'      ihy: ',ihy,ymin,ymax
            print*,'           '
            print*,' (the hash table is overfull!) '
            print*,' (you can avoid this by upping _HASHDEP_) '
            print*,'           '
            print*,'           '
            stop
         endif
         hash(ihx,ihy,nash(ihx,ihy)) = N
      enddo

      hhist = 0
      do ihx = 1, _HASHDIM_
         do ihy = 1, _HASHDIM_
            hhist(nash(ihx,ihy)+1) = hhist(nash(ihx,ihy)+1)+1
         enddo
      enddo

      NFND = 0

! Inverse transformations
      DELTA = AG*DG-BG*CG
      GA =  DG/DELTA
      GB = -BG/DELTA
      GC = -CG/DELTA
      GD =  AG/DELTA
      
      LNKFILE = OUTDIR(1:(len_trim(OUTDIR))) //
     .          IMGFILE(1:(len_trim(IMGFILE)-4)) // 'LNK'

      
      print*,''
      write(*,'('' SAVING: '',a)') LNKFILE(1:(len_trim(LNKFILE)))
      print*,''
      
      open(50,file=LNKFILE,status='unknown')
      
      write(50,'(''# '',
     .''xc_hst yc_hst m_hst q_hst '',
     .''xr_hst yr_hst '',
     .''xhst_gaia '',
     .''yhst_gaia '',
     .''x_gaia y_gaia g_gaia '',
     .''xpm_gaia ypm_gaia '',
     .''dra_gaia ddec_gaia '',
     .''ra_gaia dec_gaia '',
     .''era_gaia edec_gaia '',
     .''pmra_gaia epmra_gaia '',
     .''pmdec_gaia epmdec_gaia '')')
      
      do N = 1, size(xpm_o)
         
         if (xpm_o(N).eq.0.d0.and.ypm_o(N).eq.0.d0) then
            write(50,150)
     .     0.0d0, 0.0d0, 0.0d0, 0.d0,                                  ! xc_hst, yc_hst, m_hst, q_hst
     .     0.0d0, 0.0d0,                                               ! xr_hst, yr_hst
     .     0.0d0,                                                      ! xhst_gaia
     .     0.0d0,                                                      ! yhst_gaia
     .     x_o(N), y_o(N), mg_o(N),                                    ! x_gaia, y_gaia, g_gaia
     .     xpm_o(N), ypm_o(N),                                         ! xpm_gaia, ypm_gaia,
     .     dra_o(N), dde_o(N),                                         ! dra_gaia, ddec_gaia
     .     ra_o(N), de_o(N),                                           ! ra_gaia, dec_gaia
     .     era_o(N)/1000.d0, ede_o(N)/1000.d0,                         ! era_gaia, edec_gaia
     .     mura_o(N), emura_o(N),                                      ! pmra_gaia, epmra_gaia
     .     mude_o(N), emude_o(N)                                       ! pmdec_gaia, epmdec_gaia
            cycle
         endif
         
         xr = GA*(xpm_o(N)-Wo)+GB*(ypm_o(N)-Zo)+Xo
         yr = GC*(xpm_o(N)-Wo)+GD*(ypm_o(N)-Zo)+Yo
         ihxc = 1 + (_HASHDIM_-1)*(xr-xmin)/(xmax-xmin)
         ihyc = 1 + (_HASHDIM_-1)*(yr-ymin)/(ymax-ymin)
         NSTARu = 0
         if (xr.lt.xmin-5.or.xr.gt.xmax+5.or.
     .       yr.lt.ymin-5.or.yr.gt.ymax+5) goto 999

         dmin = MINRAD
         
         do ihx = max(1,ihxc-5),min(_HASHDIM_,ihxc+5)
            do ihy = max(1,ihyc-5),min(_HASHDIM_,ihyc+5)
               do nhh = 1, nash(ihx,ihy)
                  NSTAR = hash(ihx,ihy,nhh)
                  if (abs(xr-xchsto(NSTAR)).lt.9.and.
     .                abs(yr-ychsto(NSTAR)).lt.9) then
                     duse = sqrt((xr-xchsto(NSTAR))**2 +
     .                           (yr-ychsto(NSTAR))**2)
                     if (duse.lt.dmin) then
                        NSTARu = NSTAR
                        dmin = duse
                     endif
                  endif
               enddo
            enddo
         enddo
 999     continue
         Ou = N
         if (NSTARu.eq.0) write(50,150)
     .     0.0d0, 0.0d0, 0.0d0, 0.d0,                                    ! xc_hst, yc_hst, m_hst, q_hst
     .     0.0d0, 0.0d0,                                                 ! xr_hst, yr_hst
     .     0.0d0,                                                        ! xhst_gaia
     .     0.0d0,                                                        ! yhst_gaia
     .     x_o(Ou), y_o(Ou), mg_o(Ou),                                   ! x_gaia, y_gaia, g_gaia
     .     xpm_o(Ou), ypm_o(Ou),                                         ! xpm_gaia, ypm_gaia,
     .     dra_o(Ou), dde_o(Ou),                                         ! dra_gaia, ddec_gaia
     .     ra_o(Ou), de_o(Ou),                                           ! ra_gaia, dec_gaia
     .     era_o(Ou)/1000.d0, ede_o(Ou)/1000.d0,                         ! era_gaia, edec_gaia
     .     mura_o(Ou), emura_o(Ou),                                      ! pmra_gaia, epmra_gaia
     .     mude_o(Ou), emude_o(Ou)                                       ! pmdec_gaia, epmdec_gaia
         if (NSTARu.ne.0) write(50,150)
     .     xchsto(NSTARu), ychsto(NSTARu), mhsto(NSTARu), qhsto(NSTARu), ! xc_hst, yc_hst, m_hst, q_hst
     .     xrhsto(NSTARu), yrhsto(NSTARu),                               ! xr_hst, yr_hst
     .       (AG*(xchsto(NSTARu)-Xo)+BG*(ychsto(NSTARu)-Yo)+Wo),         ! xhst_gaia
     .       (CG*(xchsto(NSTARu)-Xo)+DG*(ychsto(NSTARu)-Yo)+Zo),         ! yhst_gaia
     .     x_o(Ou), y_o(Ou), mg_o(Ou),                                   ! x_gaia, y_gaia, g_gaia
     .     xpm_o(Ou), ypm_o(Ou),                                         ! xpm_gaia, ypm_gaia,
     .     dra_o(Ou), dde_o(Ou),                                         ! dra_gaia, ddec_gaia
     .     ra_o(Ou), de_o(Ou),                                           ! ra_gaia, dec_gaia
     .     era_o(Ou)/1000.d0, ede_o(Ou)/1000.d0,                         ! era_gaia, edec_gaia
     .     mura_o(Ou), emura_o(Ou),                                      ! pmra_gaia, epmra_gaia
     .     mude_o(Ou), emude_o(Ou)                                       ! pmdec_gaia, epmdec_gaia
      enddo

      write(50,'(''# '',
     .''(1) (2) (3) (4) '',
     .''(5) (6) '',
     .''(7) '',
     .''(8) '',
     .''(9) (10) (11) '',
     .''(12) (13) '',
     .''(14) (15) '',
     .''(16) (17) '',
     .''(18) (19) '',
     .''(20) (21) '',
     .''(22) (23) '')')
      write(50,'(''# '',
     .''hst_pixel hst_pixel mag - '',
     .''hst_pixel hst_pixel '',
     .''ref_pixel '',
     .''ref_pixel '',
     .''ref_pixel ref_pixel mag '',
     .''ref_pixel ref_pixel '',
     .''mas mas '',
     .''deg deg '',
     .''mas mas '',
     .''mas/yr mas/yr '',
     .''mas/yr mas/yr '')')
      write(50,'(''# '',
     .''xc_hst yc_hst m_hst q_hst '',
     .''xr_hst yr_hst '',
     .''xhst_gaia '',
     .''yhst_gaia '',
     .''x_gaia y_gaia g_gaia '',
     .''xpm_gaia ypm_gaia '',
     .''dra_gaia ddec_gaia '',
     .''ra_gaia dec_gaia '',
     .''era_gaia edec_gaia '',
     .''pmra_gaia epmra_gaia '',
     .''pmdec_gaia epmdec_gaia '')')
      
      close(50)

      deallocate(xgaia,ygaia,mgaia,ngaia)

 150  format(f12.4,1x,f12.4,1x,f12.4,1x,f12.4,1x,                        ! xc_hst, yc_hst, m_hst, q_hst
     .      f12.4,1x,f12.4,1x,                                           ! xr_hst, yr_hst
     .      f12.4,1x,                                                    ! xhst_gaia
     .      f12.4,1x,                                                    ! yhst_gaia
     .      f12.4,1x,f12.4,1x,f12.4,1x,                                  ! x_gaia, y_gaia, g_gaia
     .      f12.4,1x,f12.4,1x,                                           ! xpm_gaia, ypm_gaia
     .      f16.5,1x,f16.5,1x,                                           ! dra_gaia, ddec_gaia
     .      f14.10,1x,f14.10,1x,                                         ! ra_gaia, dec_gaia
     .      f14.5,1x,f14.5,1x,                                           ! era_gaia, edec_gaia
     .      f10.4,1x,f10.4,1x,                                           ! pmra_gaia, epmra_gaia
     .      f10.4,1x,f10.4)                                              ! pmdec_gaia, epmdec_gaia


      if (AMP) then
! It's easier and faster if I read the HST data again from the LNK file.
! This way, all stars are already ordered as in the Gaia catalog.         
! If a star is not present, I don't care...
      
         deallocate(xchsto)
         deallocate(ychsto)
         deallocate(mhsto)
         deallocate(qhsto)
         deallocate(xrhsto)
         deallocate(yrhsto)
         deallocate(nhsto)
         
         print*,''
         print*,''
         write(*,'('' ---------------------------'')')
         write(*,'(''|                           |'')')
         write(*,'(''| AMPLIFIER TRANSFORMATIONS |'')')
         write(*,'(''|                           |'')')
         write(*,'('' ---------------------------'')')
         print*,''
         
         allocate(AG_K(4),BG_K(4),CG_K(4),DG_K(4))
         allocate(Xo_K(4),Yo_K(4),Wo_K(4),Zo_K(4))
         AG_K = 0.d0
         BG_K = 0.d0
         CG_K = 0.d0
         DG_K = 0.d0
         Xo_K = 0.d0
         Yo_K = 0.d0
         Wo_K = 0.d0
         Zo_K = 0.d0

         NSTARs = size(xpm_o)
         
         allocate(xchsto(NSTARs))
         allocate(ychsto(NSTARs))
         allocate(mhsto(NSTARs))
         allocate(qhsto(NSTARs))
         allocate(xrhsto(NSTARs))
         allocate(yrhsto(NSTARs))
         allocate(nhsto(NSTARs))
         allocate(khsto(NSTARs))
         xchsto = 0.d0
         ychsto = 0.d0
         mhsto = 0.d0
         qhsto = 0.d0      
         xrhsto = 0.d0
         yrhsto = 0.d0
         nhsto = 0
         khsto = 0
         
         open(50,file=LNKFILE,status='old')
         NSTARs = 0
         do
            read(50,'(200a)',iostat=STAT) INSTRING
            if (IS_IOSTAT_END(STAT)) exit
            if (INSTRING(1:1).eq.'#') cycle
            NSTARs = NSTARs + 1
            read(INSTRING,*) xchsto(NSTARs), ychsto(NSTARs),
     .                       mhsto(NSTARs), qhsto(NSTARs),
     .                       xrhsto(NSTARs), yrhsto(NSTARs)
            nhsto(NSTARs) = NSTARs
            if (xrhsto(NSTARs).lt.2048.0d0.and.
     .          yrhsto(NSTARs).lt.2048.0d0) khsto(NSTARs) = 1
            if (xrhsto(NSTARs).lt.2048.0d0.and.
     .          yrhsto(NSTARs).ge.2048.0d0) khsto(NSTARs) = 2
            if (xrhsto(NSTARs).ge.2048.0d0.and.
     .          yrhsto(NSTARs).lt.2048.0d0) khsto(NSTARs) = 3
            if (xrhsto(NSTARs).ge.2048.0d0.and.
     .          yrhsto(NSTARs).ge.2048.0d0) khsto(NSTARs) = 4
         enddo
         close(50)

         allocate(xhst_amp(NSTARs),yhst_amp(NSTARs))
         allocate(xhst_loc(NSTARs),yhst_loc(NSTARs))
         xhst_amp = 0.d0
         yhst_amp = 0.d0
         xhst_loc = 0.d0
         yhst_loc = 0.d0

         allocate(star_id(NSTARs))
         do NSTAR = 1, NSTARs
            star_id(NSTAR) = NSTAR
         enddo

         allocate(okflag(NSTARs))
         okflag = .false.

! Compute amplifier-based transformations
         do k = 1, 4
            okflag = xpm_o.gt.0.d0.and.ypm_o.gt.0.d0.and.   ! Gaia conditions
     .               usegaia.eq.1.and.
     .               xrhsto.gt.0.d0.and.yrhsto.gt.0.d0.and. ! HST conditions
     .               mhsto.lt.mlim.and.qhsto.lt.qlim.and. 
     .               ((USESAT.and.qhsto.ge.0.d0).or.
     .           (.not.USESAT.and.qhsto.gt.0.d0)).and.
     .               khsto.eq.k

            print*,''
            write(*,'(''  AMPLIFIER '',i1)') k
            
            if (count(okflag).lt.NAMP) then
               print*,''
               write(*,'('' OK, ONLY '',i6,'' STARS ARE PRESENT ('',i6,
     .'' REQUIRED): TRANSFORMATIONS NOT COMPUTED'')')
     .              count(okflag), NAMP
               cycle
            endif
            
            xgaia = pack(xpm_o,okflag)
            ygaia = pack(ypm_o,okflag)

            xhst = pack(xchsto,okflag)
            yhst = pack(ychsto,okflag)

            NMATs = size(xgaia)
            
            call glob_fit6nrDP(xhst, yhst, xgaia, ygaia, NMATs,
     .           AG_K(k), BG_K(k), CG_K(k), DG_K(k),
     .           Xo_K(k), Yo_K(k), Wo_K(k), Zo_K(k))

            write(*,'(''            STARS FOUND IN COMMON: '',i6)')
     .           NMATs
            print*,''
            write(*,'(''                               AG: ''f14.11)')
     .           AG_K(k)
            write(*,'(''                               BG: ''f14.11)')
     .            BG_K(k)
            write(*,'(''                               CG: ''f14.11)')
     .            CG_K(k)
            write(*,'(''                               DG: ''f14.11)')
     .            DG_K(k)
            write(*,'(''                               Xo: ''f14.4)')
     .            Xo_K(k)
            write(*,'(''                               Yo: ''f14.4)')
     .            Yo_K(k)
            write(*,'(''                               Wo: ''f14.4)')
     .            Wo_K(k)
            write(*,'(''                               Zo: ''f14.4)')
     .            Zo_K(k)
            print*,''

            deallocate(xgaia,ygaia,xhst,yhst)
         enddo
         
! Begin local transformation
         do NSTAR = 1, NSTARs

            if (xrhsto(NSTAR).eq.0.d0.and.yrhsto(NSTAR).eq.0.d0.and.  ! Star not present in HST
     .          mhsto(NSTAR).eq.0.d0.and.qhsto(NSTAR).eq.0.d0) cycle

            xhst_amp(NSTAR) =
     .           AG_K(khsto(NSTAR))*(xchsto(NSTAR)-Xo_K(khsto(NSTAR))) +
     .           BG_K(khsto(NSTAR))*(ychsto(NSTAR)-Yo_K(khsto(NSTAR))) +
     .           Wo_K(khsto(NSTAR))
            yhst_amp(NSTAR) =
     .           CG_K(khsto(NSTAR))*(xchsto(NSTAR)-Xo_K(khsto(NSTAR))) +
     .           DG_K(khsto(NSTAR))*(ychsto(NSTAR)-Yo_K(khsto(NSTAR))) +
     .           Zo_K(khsto(NSTAR))

         enddo

         LNKFILE = OUTDIR(1:(len_trim(OUTDIR))) //
     .        IMGFILE(1:(len_trim(IMGFILE)-4)) // 'AMP'

         print*,''
         write(*,'('' SAVING: '',a)') LNKFILE(1:(len_trim(LNKFILE)))
         print*,''
      
         open(50,file=LNKFILE,status='unknown')
         write(50,'(''# xhst_amp_gaia yhst_amp_gaia amplifier'')')
         do NSTAR = 1, NSTARs
            write(50,987) xhst_amp(NSTAR), yhst_amp(NSTAR), khsto(NSTAR)
         enddo
         write(50,'(''# (1) (2) (3)'')')
         write(50,'(''# hst_pixel hst_pixel - '')')
         write(50,'(''# xhst_amp_gaia yhst_amp_gaia amplifier'')')
         close(50)
 987     format(f12.4,1x,f12.4,1x,i1)
         
      endif
      
      
      print*,''
      print*,''
      print*,''

      stop

    
      end program xym2pm_Gaia


       






c-----------------------------------------------
c 
c This routine will take an (ra,dec) and a 
c tangent-point (ra0,dec0) and will compute
c the y coordinate in the tangent plane.  The 
c tangent-plane y-axis is directed along +DEC.
c The units for y will be degrees, just lilke
c those for RA and DEC.
c
      real*8 function rd2y(r,d,r0,d0)
      implicit none

      real*8 r, d
      real*8 r0,d0

      real*8 cosra, sinra
      real*8 cosde, sinde
      real*8 cosd0, sind0
      real*8 rrrr
      real*8 yrad, PI_314
      real*8 x, y, z
      real*8 xx,yy,zz

      PI_314 = 4.d0*atan(1.0d0)

      cosra = cos((r-r0)*PI_314/180.0d0)
      sinra = sin((r-r0)*PI_314/180.0d0)
      cosde = cos(d *PI_314/180.0d0)
      sinde = sin(d *PI_314/180.0d0)
      cosd0 = cos(d0*PI_314/180.0d0)
      sind0 = sin(d0*PI_314/180.0d0)

      rrrr = sind0*sinde + cosd0*cosde*cosra

      yrad = (cosd0*sinde-sind0*cosde*cosra)/rrrr
      rd2y = yrad*180.0d0/PI_314

      x  = cosde*cos(r *PI_314/180.0d0)
      y  = cosde*sin(r *PI_314/180.0d0)
      z  = sinde
      xx = cosd0*cos(r0*PI_314/180.0d0)
      yy = cosd0*sin(r0*PI_314/180.0d0)
      zz = sind0

      if (x*xx + y*yy + z*zz.lt.0.0d0) rd2y = 90.0d0

      return
      end


c-----------------------------------------------
c 
c This routine will take an (ra,dec) and a 
c tangent-point (ra0,dec0) and will compute
c the x coordinate in the tangent plane.  The 
c tangent-plane x-axis is directed along -RA.
c The units for x will be degrees, just lilke
c those for RA and DEC.
c
      real*8 function rd2x(r,d,r0,d0)
      implicit none

      real*8 r, d
      real*8 r0,d0

      real*8 cosra, sinra
      real*8 cosde, sinde
      real*8 cosd0, sind0
      real*8 rrrr
      real*8 xrad, PI_314
      real*8 x, y, z
      real*8 xx,yy,zz
      
      PI_314 = 4.d0*atan(1.0d0)

      cosra = cos((r-r0)*PI_314/180.0d0)
      sinra = sin((r-r0)*PI_314/180.0d0)

      cosde = cos(d *PI_314/180.0d0)
      sinde = sin(d *PI_314/180.0d0)
      cosd0 = cos(d0*PI_314/180.0d0)
      sind0 = sin(d0*PI_314/180.0d0)
  
      rrrr = sind0*sinde + cosd0*cosde*cosra


      xrad = cosde*sinra/rrrr
      rd2x = xrad*180.0d0/PI_314
  
      x  = cosde*cos(r *PI_314/180.d0)
      y  = cosde*sin(r *PI_314/180.d0)
      z  = sinde
      xx = cosd0*cos(r0*PI_314/180.d0)
      yy = cosd0*sin(r0*PI_314/180.d0)
      zz = sind0

      if (x*xx + y*yy + z*zz.lt.0.d0) rd2x = 90.0d0

      return
      end





      SUBROUTINE QSORTI (ORD,N,A)
C
C==============SORTS THE ARRAY A(I),I=1,2,...,N BY PUTTING THE
C   ASCENDING ORDER VECTOR IN ORD.  THAT IS ASCENDING ORDERED A
C   IS A(ORD(I)),I=1,2,...,N; DESCENDING ORDER A IS A(ORD(N-I+1)),
C   I=1,2,...,N .  THIS SORT RUNS IN TIME PROPORTIONAL TO N LOG N .
C
C
C     ACM QUICKSORT - ALGORITHM #402 - IMPLEMENTED IN FORTRAN 66 BY
C                                 WILLIAM H. VERITY, WHV@PSUVM.PSU.EDU
C                                 CENTER FOR ACADEMIC COMPUTING
C                                 THE PENNSYLVANIA STATE UNIVERSITY
C                                 UNIVERSITY PARK, PA.  16802
C
      IMPLICIT INTEGER (A-Z)
C
      DIMENSION ORD(N),POPLST(2,20)
C
C     TO SORT DIFFERENT INPUT TYPES, CHANGE THE FOLLOWING
C     SPECIFICATION STATEMENTS; FOR EXAMPLE, FOR FORTRAN CHARACTER
C     USE THE FOLLOWING:  CHARACTER *(*) A(N)
C
      REAL A(N)
      REAL X,XX,Z,ZZ,Y
C
      NDEEP=0
      U1=N
      L1=1
      DO 1  I=1,N
    1 ORD(I)=I
    2 IF (U1.LE.L1) RETURN
C
    3 L=L1
      U=U1
C
C PART
C
    4 P=L
      Q=U
C     FOR CHARACTER SORTS, THE FOLLOWING 3 STATEMENTS WOULD BECOME
C     X = ORD(P)
C     Z = ORD(Q)
C     IF (A(X) .LE. A(Z)) GO TO 2
C
C     WHERE "CLE" IS A LOGICAL FUNCTION WHICH RETURNS "TRUE" IF THE
C     FIRST ARGUMENT IS LESS THAN OR EQUAL TO THE SECOND, BASED ON "LEN"
C     CHARACTERS.
C
      X=A(ORD(P))
      Z=A(ORD(Q))
      IF (X.LE.Z) GO TO 5
      Y=X
      X=Z
      Z=Y
      YP=ORD(P)
      ORD(P)=ORD(Q)
      ORD(Q)=YP
    5 IF (U-L.LE.1) GO TO 15
      XX=X
      IX=P
      ZZ=Z
      IZ=Q
C
C LEFT
C
    6 P=P+1
      IF (P.GE.Q) GO TO 7
      X=A(ORD(P))
      IF (X.GE.XX) GO TO 8
      GO TO 6
    7 P=Q-1
      GO TO 13
C
C RIGHT
C
    8 Q=Q-1
      IF (Q.LE.P) GO TO 9
      Z=A(ORD(Q))
      IF (Z.LE.ZZ) GO TO 10
      GO TO 8
    9 Q=P
      P=P-1
      Z=X
      X=A(ORD(P))
C
C DIST
C
   10 IF (X.LE.Z) GO TO 11
      Y=X
      X=Z
      Z=Y
      IP=ORD(P)
      ORD(P)=ORD(Q)
      ORD(Q)=IP
   11 IF (X.LE.XX) GO TO 12
      XX=X
      IX=P
   12 IF (Z.GE.ZZ) GO TO 6
      ZZ=Z
      IZ=Q
      GO TO 6
C
C OUT
C
   13 CONTINUE
      IF (.NOT.(P.NE.IX.AND.X.NE.XX)) GO TO 14
      IP=ORD(P)
      ORD(P)=ORD(IX)
      ORD(IX)=IP
   14 CONTINUE
      IF (.NOT.(Q.NE.IZ.AND.Z.NE.ZZ)) GO TO 15
      IQ=ORD(Q)
      ORD(Q)=ORD(IZ)
      ORD(IZ)=IQ
   15 CONTINUE
      IF (U-Q.LE.P-L) GO TO 16
      L1=L
      U1=P-1
      L=Q+1
      GO TO 17
   16 U1=U
      L1=Q+1
      U=P-1
   17 CONTINUE
      IF (U1.LE.L1) GO TO 18
C
C START RECURSIVE CALL
C
      NDEEP=NDEEP+1
      POPLST(1,NDEEP)=U
      POPLST(2,NDEEP)=L
      GO TO 3
   18 IF (U.GT.L) GO TO 4
C
C POP BACK UP IN THE RECURSION LIST
C
      IF (NDEEP.EQ.0) GO TO 2
      U=POPLST(1,NDEEP)
      L=POPLST(2,NDEEP)
      NDEEP=NDEEP-1
      GO TO 18
C
C END SORT
C END QSORT
C
      END




c------------------------------------------------
c
c same as rclip, but for double precision numbers
c
      real*8 function dclip(dval,dlo,dhi)
      implicit none
 
      real*8 dval
      real*8 dlo
      real*8 dhi

      dclip = dval
      if (dclip.gt.dhi) dclip = dhi
      if (dclip.lt.dlo) dclip = dlo
      if (.not.(dclip.lt.dhi).and.
     .    .not.(dclip.gt.dlo)) dclip = dhi
    
      return
      end






! ML -> This routine contains the coordinates of a list of objects.
!       At the moment, there are only GCs.
      subroutine object2radec(INPUT, RA, DEC)
      implicit none
      
      character*40 INPUT, NAME, RA, DEC

      RA = '0'
      DEC = '0'

      call upper2lower(INPUT,NAME)

      if (NAME(1:10).eq.'objectlist') then
         print*,''
         print*,'HARD-CODED OBJECT LIST:'
         print*,''
         print*,'ngc104,47Tuc'
         print*,'ngc288'                               
         print*,'ngc362'                               
         print*,'ngc1261'                              
         print*,'palomar1'                             
         print*,'palomar2'                             
         print*,'ngc1851'                              
         print*,'ngc2298'                              
         print*,'ngc2808'                              
         print*,'E3'                                   
         print*,'ngc3201'                              
         print*,'ngc4147'                              
         print*,'ngc4590,m68'       
         print*,'ngc4833'                              
         print*,'ngc5024,m53'       
         print*,'ngc5053'                              
         print*,'ngc5139'                              
         print*,'ngc5272,m3'      
         print*,'ngc5286'                              
         print*,'ngc5466'                              
         print*,'ngc5904,m5'        
         print*,'ngc5927'                              
         print*,'ngc5986'                              
         print*,'lynga7 '                              
         print*,'ngc6093,m80'       
         print*,'ngc6101'                              
         print*,'ngc6121,m4'        
         print*,'ngc6144'                              
         print*,'ngc6171,m107'
         print*,'ngc6205,m13'       
         print*,'ngc6218,m12'       
         print*,'ngc6254,m10'       
         print*,'ngc6304'                              
         print*,'ngc6341,m92'       
         print*,'ngc6352'                              
         print*,'ngc6362'                              
         print*,'ngc6366'                              
         print*,'ngc6388'                              
         print*,'ngc6397'                              
         print*,'ngc6441'                              
         print*,'ngc6496'                              
         print*,'ngc6535'                              
         print*,'ngc6541'                              
         print*,'ngc6584'                              
         print*,'ngc6624'                              
         print*,'ngc6637,m69'       
         print*,'ngc6652'                              
         print*,'ngc6656,m22'       
         print*,'ngc6681,m70'       
         print*,'ngc6715,m54'       
         print*,'ngc6717,palomar9'
         print*,'ngc6723'                              
         print*,'ngc6752'                              
         print*,'ngc6779,m56'       
         print*,'terzan7'                              
         print*,'arp2'                                 
         print*,'ngc6809,m55'
         print*,'terzan8'                              
         print*,'ngc6838,m71'
         print*,'ngc6934'                              
         print*,'ngc6981,m72'
         print*,'ngc7078,m15'
         print*,'ngc7089,m2'
         print*,'ngc7099,m30'
         print*,'palomar12'                            
         print*,''
         stop
      endif

      if (NAME(1:6).eq.'ngc104'.or.NAME(1:5).eq.'47tuc')      
     .     RA='RA_CEN=00:24:05.71' 
      if (NAME(1:6).eq.'ngc288')                               
     .     RA='RA_CEN=00:52:45.24' 
      if (NAME(1:6).eq.'ngc362')                              
     .     RA='RA_CEN=01:03:14.26' 
      if (NAME(1:7).eq.'ngc1261')                              
     .     RA='RA_CEN=03:12:16.21' 
      if (NAME(1:8).eq.'palomar1')                             
     .     RA='RA_CEN=03:33:20.04'
      if (NAME(1:8).eq.'palomar2')                             
     .     RA='RA_CEN=04:46:05.91'
      if (NAME(1:7).eq.'ngc1851')                              
     .     RA='RA_CEN=05:14:06.76' 
      if (NAME(1:7).eq.'ngc2298')                             
     .     RA='RA_CEN=06:48:59.41' 
      if (NAME(1:7).eq.'ngc2808')                              
     .     RA='RA_CEN=09:12:03.10' 
      if (NAME(1:2).eq.'E3')                                   
     .     RA='RA_CEN=09:20:57.07' 
      if (NAME(1:7).eq.'ngc3201')                              
     .     RA='RA_CEN=10:17:36.82'
      if (NAME(1:7).eq.'ngc4147')                              
     .     RA='RA_CEN=12:10:06.30'
      if (NAME(1:7).eq.'ngc4590'.or.NAME(1:3).eq.'m68')       
     .     RA='RA_CEN=12:39:27.98'
      if (NAME(1:7).eq.'ngc4833')                              
     .     RA='RA_CEN=12:59:33.92'
      if (NAME(1:7).eq.'ngc5024'.or.NAME(1:3).eq.'m53')       
     .     RA='RA_CEN=13:12:55.25'
      if (NAME(1:7).eq.'ngc5053')                              
     .     RA='RA_CEN=13:16:27.09'
      if (NAME(1:7).eq.'ngc5139')                              
     .     RA='RA_CEN=13:26:47.24'
      if (NAME(1:7).eq.'ngc5272'.or.(NAME(1:2).eq.'m3'.and.
     .     NAME(3:3).eq.' '))
     .     RA='RA_CEN=13:42:11.62'
      if (NAME(1:7).eq.'ngc5286')                              
     .     RA='RA_CEN=13:46:26.81'
      if (NAME(1:7).eq.'ngc5466')                              
     .     RA='RA_CEN=14:05:27.29'
      if (NAME(1:7).eq.'ngc5904'.or.(NAME(1:2).eq.'m5'.and.
     .     NAME(3:3).eq.' '))
     .     RA='RA_CEN=15:18:33.22'
      if (NAME(1:7).eq.'ngc5927')                              
     .     RA='RA_CEN=15:28:00.69'
      if (NAME(1:7).eq.'ngc5986')                              
     .     RA='RA_CEN=15:46:03.00'
      if (NAME(1:6).eq.'lynga7')                              
     .     RA='RA_CEN=16:11:03.65'
      if (NAME(1:7).eq.'ngc6093'.or.NAME(1:3).eq.'m80')
     .     RA='RA_CEN=16:17:02.41'
      if (NAME(1:7).eq.'ngc6101')                             
     .     RA='RA_CEN=16:25:48.12' 
      if (NAME(1:7).eq.'ngc6121'.or.NAME(1:2).eq.'m4')        
     .     RA='RA_CEN=16:23:35.22'
      if (NAME(1:7).eq.'ngc6144')                              
     .     RA='RA_CEN=16:27:13.86'
      if (NAME(1:7).eq.'ngc6171'.or.NAME(1:4).eq.'m107')      
     .     RA='RA_CEN=16:32:31.86'
      if (NAME(1:7).eq.'ngc6205'.or.NAME(1:3).eq.'m13')       
     .     RA='RA_CEN=16:41:41.24'
      if (NAME(1:7).eq.'ngc6218'.or.NAME(1:3).eq.'m12')       
     .     RA='RA_CEN=16:47:14.18'
      if (NAME(1:7).eq.'ngc6254'.or.(NAME(1:3).eq.'m10'.and.
     .     NAME(4:4).eq.' '))
     .     RA='RA_CEN=16:57:09.05'
      if (NAME(1:7).eq.'ngc6304')                              
     .     RA='RA_CEN=17:14:32.25'
      if (NAME(1:7).eq.'ngc6341'.or.NAME(1:3).eq.'m92')       
     .     RA='RA_CEN=17:17:07.39'
      if (NAME(1:7).eq.'ngc6352')                              
     .     RA='RA_CEN=17:25:29.11'
      if (NAME(1:7).eq.'ngc6362')                              
     .     RA='RA_CEN=17:31:54.99' 
      if (NAME(1:7).eq.'ngc6366')                              
     .     RA='RA_CEN=17:27:44.24'
      if (NAME(1:7).eq.'ngc6388')                              
     .     RA='RA_CEN=17:36:17.23'
      if (NAME(1:7).eq.'ngc6397')                              
     .     RA='RA_CEN=17:40:42.09' 
      if (NAME(1:7).eq.'ngc6441')                              
     .     RA='RA_CEN=17:50:13.06'
      if (NAME(1:7).eq.'ngc6496')                              
     .     RA='RA_CEN=17:59:03.68' 
      if (NAME(1:7).eq.'ngc6535')                              
     .     RA='RA_CEN=18:03:50.51'
      if (NAME(1:7).eq.'ngc6541')                              
     .     RA='RA_CEN=18:08:02.36' 
      if (NAME(1:7).eq.'ngc6584')                              
     .     RA='RA_CEN=18:18:37.60' 
      if (NAME(1:7).eq.'ngc6624')                              
     .     RA='RA_CEN=18:23:40.51'
      if (NAME(1:7).eq.'ngc6637'.or.NAME(1:3).eq.'m69')       
     .     RA='RA_CEN=18:31:23.10'
      if (NAME(1:7).eq.'ngc6652')                              
     .     RA='RA_CEN=18:35:45.63'
      if (NAME(1:7).eq.'ngc6656'.or.NAME(1:3).eq.'m22')       
     .     RA='RA_CEN=18:36:23.94'
      if (NAME(1:7).eq.'ngc6681'.or.NAME(1:3).eq.'m70')       
     .     RA='RA_CEN=18:43:12.76'
      if (NAME(1:7).eq.'ngc6715'.or.NAME(1:3).eq.'m54')       
     .     RA='RA_CEN=18:55:03.33'
      if (NAME(1:7).eq.'ngc6717'.or.NAME(1:8).eq.'palomar9')  
     .     RA='RA_CEN=18:55:06.04'
      if (NAME(1:7).eq.'ngc6723')                              
     .     RA='RA_CEN=18:59:33.15'
      if (NAME(1:7).eq.'ngc6752')                              
     .     RA='RA_CEN=19:10:52.11' 
      if (NAME(1:7).eq.'ngc6779'.or.NAME(1:3).eq.'m56')       
     .     RA='RA_CEN=19:16:35.57'
      if (NAME(1:7).eq.'Terzan7')                              
     .     RA='RA_CEN=19:17:43.92'
      if (NAME(1:4).eq.'Arp2')                                 
     .     RA='RA_CEN=19:28:44.11'
      if (NAME(1:7).eq.'ngc6809'.or.NAME(1:3).eq.'m55')       
     .     RA='RA_CEN=19:39:59.71'
      if (NAME(1:7).eq.'terzan8')                              
     .     RA='RA_CEN=19:41:44.41'
      if (NAME(1:7).eq.'ngc6838'.or.NAME(1:3).eq.'m71')       
     .     RA='RA_CEN=19:53:46.49'
      if (NAME(1:7).eq.'ngc6934')                              
     .     RA='RA_CEN=20:34:11.37'
      if (NAME(1:7).eq.'ngc6981'.or.NAME(1:3).eq.'m72')       
     .     RA='RA_CEN=20:53:27.70'
      if (NAME(1:7).eq.'ngc7078'.or.NAME(1:3).eq.'m15')       
     .     RA='RA_CEN=21:29:58.33'
      if (NAME(1:7).eq.'ngc7089'.or.(NAME(1:2).eq.'m2'.and.
     .     NAME(3:3).eq.' '))
     .     RA='RA_CEN=21:33:27.02'
      if (NAME(1:7).eq.'ngc7099'.or.NAME(1:3).eq.'m30')       
     .     RA='RA_CEN=21:40:22.12'
      if (NAME(1:9).eq.'palomar12')                            
     .     RA='RA_CEN=21:46:38.84'

      if (NAME(1:6).eq.'ngc104'.or.NAME(1:5).eq.'47tuc')      
     .     DEC='DE_CEN=-72:04:52.7' 
      if (NAME(1:6).eq.'ngc288')                               
     .     DEC='DE_CEN=-26:34:57.4' 
      if (NAME(1:6).eq.'ngc362')                               
     .     DEC='DE_CEN=-70:50:55.6' 
      if (NAME(1:7).eq.'ngc1261')                              
     .     DEC='DE_CEN=-55:12:58.4' 
      if (NAME(1:8).eq.'palomar1')                             
     .     DEC='DE_CEN=+79:34:51.8'
      if (NAME(1:8).eq.'palomar2')                             
     .     DEC='DE_CEN=+31:22:53.4'
      if (NAME(1:7).eq.'ngc1851')                              
     .     DEC='DE_CEN=-40:02:47.6' 
      if (NAME(1:7).eq.'ngc2298')                              
     .     DEC='DE_CEN=-36:00:19.1' 
      if (NAME(1:7).eq.'ngc2808')                              
     .     DEC='DE_CEN=-64:51:48.6' 
      if (NAME(1:2).eq.'E3')                                   
     .     DEC='DE_CEN=-77:16:54.8' 
      if (NAME(1:7).eq.'ngc3201')                              
     .     DEC='DE_CEN=-46:24:44.9'
      if (NAME(1:7).eq.'ngc4147')                              
     .     DEC='DE_CEN=+18:32:33.5'
      if (NAME(1:7).eq.'ngc4590'.or.NAME(1:3).eq.'m68')       
     .     DEC='DE_CEN=-26:44:38.6'
      if (NAME(1:7).eq.'ngc4833')
     .     DEC='DE_CEN=-70:52:35.4'
      if (NAME(1:7).eq.'ngc5024'.or.NAME(1:3).eq.'m53')       
     .     DEC='DE_CEN=+18:10:05.4'
      if (NAME(1:7).eq.'ngc5053')                              
     .     DEC='DE_CEN=+17:42:00.9'
      if (NAME(1:7).eq.'ngc5139')                              
     .     DEC='DE_CEN=-47:28:46.45'
      if (NAME(1:7).eq.'ngc5272'.or.(NAME(1:2).eq.'m3'.and.
     .     NAME(3:3).eq.' '))
     .     DEC='DE_CEN=+28:22:38.2'
      if (NAME(1:7).eq.'ngc5286')                              
     .     DEC='DE_CEN=-51:22:27.3'
      if (NAME(1:7).eq.'ngc5466')                              
     .     DEC='DE_CEN=+28:32:04.0'
      if (NAME(1:7).eq.'ngc5904'.or.(NAME(1:2).eq.'m5'.and.
     .     NAME(3:3).eq.' '))
     .     DEC='DE_CEN=+02:04:51.7'
      if (NAME(1:7).eq.'ngc5927')                              
     .     DEC='DE_CEN=-50:40:22.9'
      if (NAME(1:7).eq.'ngc5986')                              
     .     DEC='DE_CEN=-37:47:11.1'
      if (NAME(1:6).eq.'lynga7')                              
     .     DEC='DE_CEN=-55:19:04.0'
      if (NAME(1:7).eq.'ngc6093'.or.NAME(1:3).eq.'m80')       
     .     DEC='DE_CEN=-22:58:33.9'
      if (NAME(1:7).eq.'ngc6101')                              
     .     DEC='DE_CEN=-72:12:07.9' 
      if (NAME(1:7).eq.'ngc6121'.or.NAME(1:2).eq.'m4')        
     .     DEC='DE_CEN=-26:31:32.7'
      if (NAME(1:7).eq.'ngc6144')                              
     .     DEC='DE_CEN=-26:01:24.6'
      if (NAME(1:7).eq.'ngc6171'.or.NAME(1:4).eq.'m107')      
     .     DEC='DE_CEN=-13:03:13.6'
      if (NAME(1:7).eq.'ngc6205'.or.NAME(1:3).eq.'m13')       
     .     DEC='DE_CEN=+36:27:35.5'
      if (NAME(1:7).eq.'ngc6218'.or.NAME(1:3).eq.'m12')       
     .     DEC='DE_CEN=-01:56:54.7'
      if (NAME(1:7).eq.'ngc6254'.or.(NAME(1:3).eq.'m10'.and.
     .     NAME(4:4).eq.' '))
     .     DEC='DE_CEN=-04:06:01.1'
      if (NAME(1:7).eq.'ngc6304')                              
     .     DEC='DE_CEN=-29:27:43.3'
      if (NAME(1:7).eq.'ngc6341'.or.NAME(1:3).eq.'m92')       
     .     DEC='DE_CEN=+43:08:09.4'
      if (NAME(1:7).eq.'ngc6352')                              
     .     DEC='DE_CEN=-48:25:19.8'
      if (NAME(1:7).eq.'ngc6362')                              
     .     DEC='DE_CEN=-67:02:54.0' 
      if (NAME(1:7).eq.'ngc6366')                              
     .     DEC='DE_CEN=-05:04:47.5'
      if (NAME(1:7).eq.'ngc6388')                              
     .     DEC='DE_CEN=-44:44:07.8'
      if (NAME(1:7).eq.'ngc6397')                              
     .     DEC='DE_CEN=-53:40:27.6' 
      if (NAME(1:7).eq.'ngc6441')                              
     .     DEC='DE_CEN=-37:03:05.2'
      if (NAME(1:7).eq.'ngc6496')                              
     .     DEC='DE_CEN=-44:15:57.4' 
      if (NAME(1:7).eq.'ngc6535')                              
     .     DEC='DE_CEN=-00:17:51.5'
      if (NAME(1:7).eq.'ngc6541')                             
     .     DEC='DE_CEN=-43:42:53.6' 
      if (NAME(1:7).eq.'ngc6584')                              
     .     DEC='DE_CEN=-52:12:56.8' 
      if (NAME(1:7).eq.'ngc6624')                              
     .     DEC='DE_CEN=-30:21:39.7'
      if (NAME(1:7).eq.'ngc6637'.or.NAME(1:3).eq.'m69')       
     .     DEC='DE_CEN=-32:20:53.1'
      if (NAME(1:7).eq.'ngc6652')                              
     .     DEC='DE_CEN=-32:59:26.6'
      if (NAME(1:7).eq.'ngc6656'.or.NAME(1:3).eq.'m22')       
     .     DEC='DE_CEN=-23:54:17.1'
      if (NAME(1:7).eq.'ngc6681'.or.NAME(1:3).eq.'m70')       
     .     DEC='DE_CEN=-32:17:31.6'
      if (NAME(1:7).eq.'ngc6715'.or.NAME(1:3).eq.'m54')       
     .     DEC='DE_CEN=-30:28:47.5'
      if (NAME(1:7).eq.'ngc6717'.or.NAME(1:8).eq.'palomar9')  
     .     DEC='DE_CEN=-22:42:05.3'
      if (NAME(1:7).eq.'ngc6723')                              
     .     DEC='DE_CEN=-36:37:56.1'
      if (NAME(1:7).eq.'ngc6752')                              
     .     DEC='DE_CEN=-59:59:04.4' 
      if (NAME(1:7).eq.'ngc6779'.or.NAME(1:3).eq.'m56')       
     .     DEC='DE_CEN=+30:11:00.5'
      if (NAME(1:7).eq.'Terzan7')                              
     .     DEC='DE_CEN=-34:39:27.8'
      if (NAME(1:4).eq.'Arp2')                                 
     .     DEC='DE_CEN=-30:21:20.3'
      if (NAME(1:7).eq.'ngc6809'.or.NAME(1:3).eq.'m55')       
     .     DEC='DE_CEN=-30:57:53.1'
      if (NAME(1:7).eq.'terzan8')                              
     .     DEC='DE_CEN=-33:59:58.1'
      if (NAME(1:7).eq.'ngc6838'.or.NAME(1:3).eq.'m71')       
     .     DEC='DE_CEN=+18:46:45.1'
      if (NAME(1:7).eq.'ngc6934')                              
     .     DEC='DE_CEN=+07:24:16.1'
      if (NAME(1:7).eq.'ngc6981'.or.NAME(1:3).eq.'m72')       
     .     DEC='DE_CEN=-12:32:14.3'
      if (NAME(1:7).eq.'ngc7078'.or.NAME(1:3).eq.'m15')       
     .     DEC='DE_CEN=+12:10:01.2'
      if (NAME(1:7).eq.'ngc7089'.or.(NAME(1:2).eq.'m2'.and.
     .     NAME(3:3).eq.' '))
     .     DEC='DE_CEN=-00:49:23.7'
      if (NAME(1:7).eq.'ngc7099'.or.NAME(1:3).eq.'m30')       
     .     DEC='DE_CEN=-23:10:47.5'
      if (NAME(1:9).eq.'palomar12')                            
     .     DEC='DE_CEN=-21:15:09.4'


      if (RA.eq.'0'.or.DEC.eq.'0') then
         print*,''
         print*,'STOP: OBJECT NOT IN THE HARD-CODED LIST'
         print*,''         
         stop
      endif

      return

      end subroutine object2radec



! ML
      subroutine upper2lower(INPUT,OUTPUT)
      implicit none

      character*40 INPUT, OUTPUT
      integer i
      
      OUTPUT = INPUT
      do i = 1, 40
         if (INPUT(i:i).eq.'A') OUTPUT(i:i)='a'
         if (INPUT(i:i).eq.'B') OUTPUT(i:i)='b'
         if (INPUT(i:i).eq.'C') OUTPUT(i:i)='c'
         if (INPUT(i:i).eq.'D') OUTPUT(i:i)='d'
         if (INPUT(i:i).eq.'E') OUTPUT(i:i)='e'
         if (INPUT(i:i).eq.'F') OUTPUT(i:i)='f'
         if (INPUT(i:i).eq.'G') OUTPUT(i:i)='g'
         if (INPUT(i:i).eq.'H') OUTPUT(i:i)='h'
         if (INPUT(i:i).eq.'I') OUTPUT(i:i)='i'
         if (INPUT(i:i).eq.'J') OUTPUT(i:i)='j'
         if (INPUT(i:i).eq.'K') OUTPUT(i:i)='k'
         if (INPUT(i:i).eq.'L') OUTPUT(i:i)='l'
         if (INPUT(i:i).eq.'M') OUTPUT(i:i)='m'
         if (INPUT(i:i).eq.'N') OUTPUT(i:i)='n'
         if (INPUT(i:i).eq.'O') OUTPUT(i:i)='o'
         if (INPUT(i:i).eq.'P') OUTPUT(i:i)='p'
         if (INPUT(i:i).eq.'Q') OUTPUT(i:i)='q'
         if (INPUT(i:i).eq.'R') OUTPUT(i:i)='r'
         if (INPUT(i:i).eq.'S') OUTPUT(i:i)='s'
         if (INPUT(i:i).eq.'T') OUTPUT(i:i)='t'
         if (INPUT(i:i).eq.'U') OUTPUT(i:i)='u'
         if (INPUT(i:i).eq.'V') OUTPUT(i:i)='v'
         if (INPUT(i:i).eq.'W') OUTPUT(i:i)='w'
         if (INPUT(i:i).eq.'X') OUTPUT(i:i)='x'
         if (INPUT(i:i).eq.'Y') OUTPUT(i:i)='y'
         if (INPUT(i:i).eq.'Z') OUTPUT(i:i)='z'
         if (INPUT(i:i).eq.' ') return
      enddo

      return

      end subroutine upper2lower













      subroutine getinfo(FILENAME,XREF,YREF,RAREF,DECREF,
     .     INSTRUME,DETECTOR,RDAT)
      implicit none

      integer i
      character filename*300
      character filename2*300
      integer NCHAR
 
      integer iargc, NARG, NARGs
      integer NIM
 
      integer iEXPT
      real    rEXPT
      integer iPA
      real    rPA
      integer iPID
      character*6 FILTSTR
      character*12 DATESTRIN
      character*9 ROOTSTR
      real*8 rRA, hRA
      real*8 rDEC, aDEC
      real*8 rRA_TARG
      real*8 rDEC_TARG
      character*16 APERNAME
 
      character*8 DATESTR
      character*8 TIMESTR
 
      integer iRAH, iRAM, iRAS, iFRA
      integer iDEG, iMIN, iSEC, iFDC
      character decsgn
      real*8  rRA0, rDC0
      real    dRA , dDC
      real    rSUNA
      real    rTEMP
      real    vaFAC
 
      integer iDAY
      integer iMON
      integer iYIR
      integer iHRS
 
      real*8  rDAT
 
      character*8  field
      character*20 stream
 
      integer ios, k
 
      character*2880 buff
 
      integer nread
 

      character*200 FILELIST 
      character*200 FILENAME_N(9999)
 
      character*11 OBJNAME
 
      integer DOY(12)
      data DOY / 000, 031, 059, 090, 120, 151,
     .           181, 212, 243, 273, 304, 334/
 
      logical DEBUG
      real rPT1, rPT2
      integer NEND
      integer NENDs
      integer NAXIS

      character*08 KEYWORD
      character*20 KEYSTREAM

      logical SHOW_REG
      logical SHOW_APER
      logical SHOW_POSTARG
      logical SHOW_KEYWORD
      logical DO_LIST

      real*8 XREF, YREF, RAREF, DECREF
      character*20 INSTRUME, DETECTOR

!      print*,'open10: ',FILENAME
         open(10,file=FILENAME,status='old',
     .     err=900,recl=2880,form='UNFORMATTED',
     .     access='DIRECT')

         NENDs = 1
         NEND  = 0
         NAXIS = 0
 
         i = 0
         nread = 0
 100     continue
         i = i + 1
         read(10,rec=i,iostat=ios) buff
         if (ios.lt.0) goto 900
         do k = 0, 35, 1
            field  = buff(k*80+01:k*80+08)
            stream = buff(k*80+11:k*80+31)
!            if (DEBUG)
!           write(*,'(i2.2,1x,i2.2,'':'',a8,'' | '',a20)')
!     .               i,k,field,stream
            if (stream(1:1).eq.'''') stream    = stream(2:20)
            if (field.eq.'EXTEND  '.and.
     .          stream(20:20).eq.'T') NENDs = 2
!            if (field.eq.'NAXIS   ') read(stream(01:20),*) NAXIS
            if (field.eq.'TIME-OBS') TIMESTR   = stream(01:08)
            if (field.eq.'DATE-OBS') DATESTRIN = stream(01:10)
!            if (field.eq.'ROOTNAME') ROOTSTR   = stream(01:10)
!            if (field.eq.'FILTNAM1'.and.
!     .          stream(1:1).eq.'F')  FILTSTR   = stream(01:06)
!            if (field.eq.'FILTNAM1'.and.
!     .          stream(1:1).eq.'F')  FILTSTR   = stream(01:06)
!            if (field.eq.'FILTER1 '.and.
!     .          stream(1:1).eq.'F')  FILTSTR = stream(01:06)
!            if (field.eq.'FILTER2 '.and.
!     .          stream(1:1).eq.'F')  FILTSTR = stream(01:06)
!            if (field.eq.'FILTER  '.and.
!     .          stream(1:1).eq.'F')  FILTSTR = stream(01:06)
!            if (field.eq.'EXPTIME ') read(stream,*,end=109) rEXPT
!            if (field.eq.'RA_TARG ') read(stream,*,end=109) rRA_TARG
!            if (field.eq.'DEC_TARG') read(stream,*,end=109) rDEC_TARG
!            if (field.eq.'PROPOSID') read(stream,*,end=109) iPID
!            if (field.eq.'PA_V3   ') read(stream,*,end=109) rPA
!            if (field.eq.'SUN_ALT ') read(stream,*,end=109) rSUNA
!            if (field.eq.'UBAY3TMP') read(stream,*,end=109) rTEMP
!            if (field.eq.'VAFACTOR') read(stream,*,end=109) vaFAC
!            if (field.eq.'CRVAL1  '.and.rRA.eq.0)
!     .                               read(stream,*,end=109) rRA
!            if (field.eq.'CRVAL2  '.and.rDEC.eq.0)
!     .                               read(stream,*,end=109) rDEC
!            if (field.eq.'POSTARG1') read(stream,*,end=109) rPT1
!            if (field.eq.'POSTARG2') read(stream,*,end=109) rPT2
!            if (field.eq.'RA_APER ') read(stream,*,end=109) rRA
!            if (field.eq.'DEC_APER') read(stream,*,end=109) rDEC
!            if (field.eq.'CRVAL1  ') read(stream,*,end=109) rRA
!            if (field.eq.'CRVAL2  ') read(stream,*,end=109) rDEC
!            if (field.eq.'APERTURE') APERNAME = stream(1:16)
!            if (field.eq.KEYWORD)    KEYSTREAM = buff(k*80+11:k*80+30)

            if (field.eq.'INSTRUME ')
     .           INSTRUME = buff(k*80+11:k*80+30)
            if (field.eq.'DETECTOR ')
     .            DETECTOR = buff(k*80+11:k*80+30)
            
            if (field.eq.'CRVAL1   ')
     .           read(buff(k*80+11:k*80+30),*) RAREF
            if (field.eq.'CRVAL2   ')
     .           read(buff(k*80+11:k*80+30),*) DECREF

            if (field.eq.'CRPIX1   ')
     .           read(buff(k*80+11:k*80+30),*) XREF
            if (field.eq.'CRPIX2   ')
     .           read(buff(k*80+11:k*80+30),*) YREF

            if (field.eq.'END     ') then
               NEND = NEND + 1
!               if (DEBUG) then
!                   print*,' '
!                   print*,'---> END CARD...'
!                   print*,'---> NEND : ',NEND
!                   print*,'---> NENDs: ',NENDs
!                   print*,'---> NAXIS: ',NAXIS
!                   print*,' '
!                   endif
               if (NEND.eq.NENDs.or.NAXIS.ne.0) goto 101
               endif
 109        continue
            enddo
         goto 100
 101     continue
         close(10)
         if (DATESTRIN(3:3).eq.'/') DATESTR(1:8) = DATESTRIN(1:8)
         if (DATESTRIN(5:5).eq.'-') then
            DATESTR(1:2) = DATESTRIN(9:10)
            DATESTR(3:3) = '/'
            DATESTR(4:5) = DATESTRIN(6:7)
            DATESTR(6:6) = '/'
            DATESTR(7:8) = DATESTRIN(3:4)
            endif
 
         if (DATESTR.ne.'DATELESS') then
            read(DATESTR(1:2),*) iDAY
            read(DATESTR(4:5),*) iMON
            read(DATESTR(7:8),*) iYIR
            endif
 
         if (TIMESTR.ne.'TIMELESS') then
            read(TIMESTR(1:2),*) iHRS
            read(TIMESTR(4:5),*) iMIN
            read(TIMESTR(7:8),*) iSEC
         endif

         rDAT = iYIR + (DOY(iMON)+iDAY)/365.0
         rDAT = rDAT + (iHRS + iMIN/60. + iSEC/60./60.)/8766.

         return

         
 900     continue
         print*,'PROBLEM!!!!!!!!!!'
         stop
         
         end
















c--------------------------------------------------------------------
c
c MIRACLE_MATCH
c
c this subroutine will take TWO photometry/astrometry lists and will
c (1) determine if there is any overlap between the two fields, and
c (2) if there is, it will give a list of stars that are the same
c     in the two images (which will allow for a local transformation)
c
c--------------------------------------------------------------------
 
      subroutine miracle_match(xin1,yin1,min1,NIN1,
     .                         xin2,yin2,min2,NIN2,
     .                         x1b,y1b,m1b,x2b,y2b,m2b,NBUOY,
     .                         NITREJ,SHOW)
      implicit none
 
      integer NIN1
      real*8    xin1(NIN1), yin1(NIN1)
      real*8    min1(NIN1)
      integer NIN2
      real*8    xin2(NIN2), yin2(NIN2)
      real*8    min2(NIN2)
      real*8    x1b(*), y1b(*), x2b(*), y2b(*)
      real*8    m1b(*), m2b(*)
      integer NBUOY
      integer NITREJ
 
      real*8 minarr, maxarr
      real*8 disc, discu
      real*8 x1bt(_NMATMX_), y1bt(_NMATMX_), m1bt(_NMATMX_)
      real*8 x2bt(_NMATMX_), y2bt(_NMATMX_), m2bt(_NMATMX_)
      integer NBT
 
      real*8 darray(25)
      data darray /99.0d0,50.00d0,25.00d0,15.0d0, 10.00d0,
     .             8.00d0, 5.00d0, 3.50d0, 2.00d0, 1.50d0,
     .             1.20d0, 1.00d0, 0.80d0, 0.60d0, 0.45d0,
     .             0.35d0, 0.25d0, 0.20d0, 0.16d0, 0.12d0,
     .             0.09d0, 0.07d0, 0.05d0, 0.04d0, 0.03d0/
      integer NITNET
 
      real*8 OX, OY
 
      integer NMAT
      integer matmeth
      integer NIT, SHOW

      NITNET = 7
 
      if (NITREJ.lt.1.or.NITREJ.gt.25) then
!         print*,'NITREJ: ',NITREJ
!         print*,'NEW MIRMAT FORMAT...'
         stop
         endif
 
!      print*,'MIRMAT:  '
!      print*,'MIRMAT: ENTER MIRMATM.F!!!  '
!      print*,'MIRMAT:  '
!      print*,'MIRMAT: MM --->     N1: ',NIN1
!      print*,'MIRMAT: MM --->     N2: ',NIN2
!      print*,'MIRMAT: MM --->  NBUOY: ',NBUOY
!      print*,'MIRMAT:   '
!      print*,'MIRMAT: INPUT PROPS:  '
!      print*,'MIRMAT:   '
!      write(*,193) 'MIRMAT:  xmin: ',minarr(xin1,NIN1),minarr(xin2,NIN2)
!      write(*,193) 'MIRMAT:  xmax: ',maxarr(xin1,NIN1),maxarr(xin2,NIN2)
!      write(*,193) 'MIRMAT:  ymin: ',minarr(yin1,NIN1),minarr(yin2,NIN2)
!      write(*,193) 'MIRMAT:  ymax: ',maxarr(yin1,NIN1),maxarr(yin2,NIN2)
!      write(*,193) 'MIRMAT:  mmin: ',minarr(min1,NIN1),minarr(min2,NIN2)
!      write(*,193) 'MIRMAT:  mmax: ',maxarr(min1,NIN1),maxarr(min2,NIN2)
!      print*,'MIRMAT:  '
!      print*,'MIRMAT:  '
      if (NBUOY.gt.0.and.NBUOY.lt.10) then
         do NMAT = 1, NBUOY
!            print*,'MIRMAT: ',NMAT,x1b(NMAT),y1b(NMAT),
!     .                         x2b(NMAT),y2b(NMAT)
            enddo
         endif
 
      matmeth = 0
 
!      if (NBUOY.eq.0) then
!         print*,' '
!         print*,'SEE IF THE STANDARD FILE "MIRMAT.HELP" IS AVAILABLE'
!         print*,' '
!         NBUOY = 0
!         open(77,file='MIRMAT.HELP',status='old',err=8)
!         NBUOY = 0
!  7      read(77,*,end=8) x1b(NBUOY+1),y1b(NBUOY+1),
!     .                    x2b(NBUOY+1),y2b(NBUOY+1)
!         NBUOY = NBUOY + 1
!         goto 7
!  8      continue
!         close(77)
!         endif

!      print*,''
 
      if (NBUOY.gt.0) goto 999
!      if (NBUOY.gt.0) then
!         print*,'  '
!         print*,'STARTING MATCHES ARE AVAILABLE FOR MIRMAT.HELP'
!         print*,'(search for matches with this starting general'
!         print*,' transformation) '
!         print*,'  '
!         print*,'    NBUOY: ',NBUOY
!         print*,'  '
!         if (NBUOY.lt.100.and.NIN1.lt.100)        ! I have found that this
!     .   call assume_orient(xin2,yin2,min2,NIN2,  ! can help find a starting
!     .                      x2b,y2b,m2b,NBUOY)    ! match if you have another
!         goto 999                                 ! image with the same orient,
!         endif                                    ! but diff offset
 
!      print*,' '
!      print*,'NO STARTING MATCHES AVAILABLE FROM MIRMAT.HELP'
!      print*,' '

      if (SHOW.eq.1) then
         print*,''
         write(*,'('' NO STARTING MATCH AVAILABLE'')')
      endif
      
c-------------------------------------------
c TEST #1
c-------------------------------------------
c
c check to see if the two starlists are related
c by  simple offset...
c
      
 111  continue
      matmeth = 1
!      print*,'MIRMAT:  '
!      print*,'MIRMAT: -------------------------------------------------'
!      print*,'MIRMAT:  '
!     print*,'MIRMAT: MM TEST1: call find_offset...'
      if (SHOW.eq.1) then
         print*,''
         write(*,'(''    1 - Simple offset:'',a1)', advance='NO')
     .        achar(13)
      endif
      call find_offset(xin1,yin1,min1,NIN1,
     .                 xin2,yin2,min2,NIN2,OX,OY,
     .                 x1b,y1b,m1b,
     .                 x2b,y2b,m2b,NBUOY)
!      print*,'MIRMAT:  '
!      print*,'MIRMAT: MM TEST1: found:  ',NBUOY
!      print*,'MIRMAT:  '
 
!      print*,'MIRMAT:                      '
!      print*,'MIRMAT:           -----------'
!      print*,'MIRMAT:           OC: OX: ',OX
!      print*,'MIRMAT:           OC: OY: ',OY
!      print*,'MIRMAT:           -----------'
!      print*,'MIRMAT:                      '
! 
!      print*,'MIRMAT:  '
      if ((NBUOY.ge.0.50*NIN1.or.
     .    (NBUOY.ge.0.50*NIN2)).and.
     .     NBUOY.ge.05) then
!           print*,'MIRMAT: '
!           print*,'MIRMAT: success in TEST1!  NBUOY = ',NBUOY
!     print*,'MIRMAT: '
         if (SHOW.eq.1) then
            write(*,'(''    1 - Simple offset: succeeded!'')')
         endif
         goto 999               ! good! we have found enough matches
      endif
      if (SHOW.eq.1) then
         write(*,'(''    1 - Simple offset: failed!'')')
      endif

c-------------------------------------------
c TEST #2
c-------------------------------------------
c
c try the standard matching procedure
c
      NBUOY = 0
 112  continue
      matmeth = 2
!      print*,'MIRMAT: -------------------------------------------------'
!      print*,'MIRMAT:  '
!      print*,'MIRMAT: MM TEST2: call MM-3D...'
!      print*,'MIRMAT:  '
      if (SHOW.eq.1) then
         print*,''
         write(*,'(''    2 - Bright permutation:'',a1)', advance='NO')
     .        achar(13)
      endif
      call miracle_match3d(xin1,yin1,min1,NIN1,
     .                     xin2,yin2,min2,NIN2,
     .                     x1b,y1b,m1b,x2b,y2b,m2b,NBUOY)
!      print*,'MIRMAT:  '
!      print*,'MIRMAT: MM TEST2: found:  ',NBUOY
!      print*,'MIRMAT:  '
      if (NBUOY.ge.10) then
!           print*,'MIRMAT: '
!           print*,'MIRMAT: success in TEST2!  NBUOY = ',NBUOY
!     print*,'MIRMAT: '
         if (SHOW.eq.1) then
            write(*,'(''    2 - Bright permutation: succeeded!'')')
         endif
         goto 999
      endif
      if (SHOW.eq.1) then
         write(*,'(''    2 - Bright permutation: failed!'')')
      endif
!      print*,'MIRMAT:  '
!      print*,'MIRMAT:  '
 

c-------------------------------------------
c TEST #3
c-------------------------------------------
c
c Use only the brightest 25 stars to find the match
c
 113  continue
      matmeth = 3
!      print*,'MIRMAT:  '
!      print*,'MIRMAT: -------------------------------------------------'
!      print*,'MIRMAT:  '
!      print*,'MIRMAT: MM TEST3:  call MM25CHEK.F (use brightest 25) '
!      print*,'MIRMAT:     NIN1: ',NIN1
!      print*,'MIRMAT:     NIN2: ',NIN2
!      print*,'MIRMAT:  '
      if (SHOW.eq.1) then
         print*,''
         write(*,'(''    3 - 25 brightest stars:'',a1)', advance='NO')
     .        achar(13)
      endif
      call miracle_match25(xin1,yin1,min1,NIN1,
     .                     xin2,yin2,min2,NIN2,
     .                     x1b,y1b,m1b,x2b,y2b,m2b,NBUOY)
!      print*,'MIRMAT:   '
!      print*,'MIRMAT: MM TEST3:  found:  ',NBUOY
!      print*,'MIRMAT:   '
      if (NBUOY.ge.10) then
!           print*,'MIRMAT: '
!           print*,'MIRMAT: success in TEST3!  NBUOY = ',NBUOY
!           print*,'MIRMAT: '
         if (SHOW.eq.1) then
            write(*,'(''    3 - 25 brightest stars: succeeded!'')')
         endif
         goto 999
      endif
      if (SHOW.eq.1) then
         write(*,'(''    3 - 25 brightest stars: failed!'')')
      endif
 
 
c-------------------------------------------
c TEST #4
c-------------------------------------------
c
c Use local triangles for the brightest stars
c
 114  continue
      matmeth = 4
!      print*,'MIRMAT:   '
!      print*,'MIRMAT: -------------------------------------------------'
!      print*,'MIRMAT:   '
!      print*,'MIRMAT: MM TEST4: call MM99CHEK.F (local triangles)'
!      print*,'MIRMAT:   '
      if (SHOW.eq.1) then
         print*,''
         write(*,'(''    4 - Local triangles:'',a1)', advance='NO')
     .        achar(13)
      endif
      call miracle_match99(xin1,yin1,min1,NIN1,
     .                     xin2,yin2,min2,NIN2,
     .                     x1b,y1b,m1b,x2b,y2b,m2b,NBUOY)
!      print*,'MIRMAT:   '
!      print*,'MIRMAT: MM TEST4: found:   ',NBUOY
!      print*,'MIRMAT:   '
      if (NBUOY.ge.10) then
!           print*,'MIRMAT: '
!           print*,'MIRMAT: success in TEST4!  NBUOY = ',NBUOY
!     print*,'MIRMAT: '
         if (SHOW.eq.1) then
            write(*,'(''    4 - Local triangles: succeeded!'')')
         endif
         goto 999
      endif
      if (SHOW.eq.1) then
         write(*,'(''    4 - Local triangles: failed!'')')
      endif
 
 
c-------------------------------------------
c TEST #5
c-------------------------------------------
c
c Do an intensive check, assuming the scale of the two
c images is the same.
c
 115  continue
      matmeth = 5
!      print*,'MIRMAT:   '
!      print*,'MIRMAT: -------------------------------------------------'
!      print*,'MIRMAT:   '
!      print*,'MIRMAT: MM TEST5: call MM5CCHEK.F (intense scale-fixed)'
!     print*,'MIRMAT:   '
      if (SHOW.eq.1) then
         print*,''
         write(*,'(''    5 - Intensive check (fixed scale):'',a1)',
     .        advance='NO') achar(13)
      endif
      call miracle_match5C(xin1,yin1,min1,NIN1,
     .                     xin2,yin2,min2,NIN2,
     .                     x1b,y1b,m1b,x2b,y2b,m2b,NBUOY)
!      print*,'MIRMAT: MM TEST5: found:  ',NBUOY
      if (NBUOY.ge.05) then
!           print*,'MIRMAT: '
!           print*,'MIRMAT: success in TEST5!  NBUOY = ',NBUOY
!     print*,'MIRMAT: '
         if (SHOW.eq.1) then
            write(*,'(''    5 - Intensive check (fixed scale):''
     .'' succeeded!'')')
         endif
         goto 999
      endif
      if (SHOW.eq.1) then
         write(*,'(''    5 - Intensive check (fixed scale): failed!'')')
      endif
 
 
c-------------------------------------------
c TEST #6
c-------------------------------------------
c
c Do an intensive check for small-number statistics,
c assuming the scale of the two images is the same,
c and also use flux ratios in the comparison.  Insist
c that positions be good to +/- 1 pixel and the
c flux ratios be the same to within 10%.
c
 116  continue
      matmeth = 6
!      print*,'MIRMAT:   '
!      print*,'MIRMAT: -------------------------------------------------'
!      print*,'MIRMAT:   '
!      print*,'MIRMAT: MM TEST6: call MM_CCHEK_SNS.F '
!      print*,'MIRMAT:           (intense scale-fixed, and '
!      print*,'MIRMAT:            optimized for small-number stats)'
!     print*,'MIRMAT:   '
      if (SHOW.eq.1) then
         print*,''
         write(*,'(''    6 - Intensive check (fixed scale and SNS):'',
     .a1)',advance='NO') achar(13)
      endif
      call miracle_match_SNS(xin1,yin1,min1,NIN1,
     .                       xin2,yin2,min2,NIN2,
     .                       x1b,y1b,m1b,x2b,y2b,m2b,NBUOY)
!      print*,'MIRMAT: MM TEST5: found:  ',NBUOY
      if (NBUOY.eq.03) then
!           print*,'MIRMAT: '
!           print*,'MIRMAT: success in TEST6!  NBUOY = ',NBUOY
!           print*,'MIRMAT: '
         if (SHOW.eq.1) then
            write(*,'(''    6 - Intensive check '',
     .''(fixed scale and SNS): succeeded!'')')
         endif
         goto 999
      endif
      if (SHOW.eq.1) then
         write(*,'(''    6 - Intensive check'',
     .'' (fixed scale and SNS): failed!'')')
      endif
 
  
!      print*,'MIRMAT:  '
!      print*,'MIRMAT: -------------------------------------------------'
!      print*,'MIRMAT:  '
!      print*,'MIRMAT:  NOT ENOUGH BUOY STARS FOUND'
!      print*,'MIRMAT:  '
!      print*,'MIRMAT:  DIED IN MMIRMATM.F '
!      print*,'MIRMAT:  '
!      print*,'MIRMAT:  RETURN NO MATCHES:   NBUOY = 0'
      NBUOY = 0

      if (SHOW.eq.1) then
         print*,''
         write(*,'('' Utterly defeated!'')')
      endif
      
      return
 
 999  continue

      
      if (SHOW.eq.1) then
         print*,''
         write(*,'('' WE HAVE A MATCH AVAILABLE!'')')
      endif

      
!      print*,'MIRMAT:  '
!      print*,'MIRMAT:  '
!      print*,'MIRMAT: MIRMAT: COPY THE MATCHES TO A TEMP ARRAY...'
!      print*,'MIRMAT:  NBUOY: ',NBUOY
!      print*,'MIRMAT:  '
      do NBT = 1, NBUOY
         x1bt(NBT) = x1b(NBT)
         y1bt(NBT) = y1b(NBT)
         m1bt(NBT) = m1b(NBT)
         x2bt(NBT) = x2b(NBT)
         y2bt(NBT) = y2b(NBT)
         m2bt(NBT) = m2b(NBT)
!         if (NBUOY.lt.25)
!     .   write(*,'('' MIRMAT: TEMP BUOY LIST -- '',i2.2,4f10.3)')
!     .                        NBT, x1bt(NBT), y1bt(NBT),
!     .                             x2bt(NBT), y2bt(NBT)
         enddo
      NBT = NBUOY
 
c-------------------------------------------------
c
c this is the starting "search" radius; disc=3 is a decent number, but if you
c have some crude measurements or are comparing ground-based against space
c based where the scales are very different, you may need to increase this
c
c the routine below will take the two raw star lists and the implied
c transormation and will search for matches within the specified radius
c "disc"
c
      if (NITREJ.lt.NITNET) NITNET = NITREJ-1
      if (NITNET.lt.1) NITNET = 1
      disc = darray(NITNET)
 
!      print*,'MIRMAT:            '
!      print*,'MIRMAT: PURIFY (remove inconsistent matches) '
!      print*,'MIRMAT:         '
!      print*,'MIRMAT:   NBT = ',NBT
!      print*,'MIRMAT:         '
      if (NBT.lt.25) then
         do NIT = 1, NBT
!            write(*,'('' MIRMAT: '',4f10.3)') x1bt(NIT),y1bt(NIT),
!     .                                        x2bt(NIT),y2bt(NIT)
            enddo
         endif
!      print*,'MIRMAT:            '

         
      if (SHOW.eq.1) then
         print*,''
         write(*,'(''    1 - Purge lists from stars not matched '',
     .''within '',f5.2,'' pix'')') disc
      endif
      
      discu = max(5.0,disc)
      discu = max(2.0,disc)
!      print*,'MIRMAT: purify     NBT: ',NBT,discu
      call purify_buoy(x1bt,y1bt,x2bt,y2bt,NBT,discu)
!      print*,'MIRMAT: purified   NBT: ',NBT,discu
 
 
 
 
!      print*,'MIRMAT: '
!      print*,'MIRMAT: '
!      print*,'MIRMAT: '
!      print*,'MIRMAT: call incorp  12: ',NIN1,NIN2
!      print*,'MIRMAT:             NBT: ',NBT
!      print*,'MIRMAT:          NITREJ: ',NITREJ
!      print*,'MIRMAT:          NITNET: ',NITNET
!      print*,'MIRMAT:            disc: ',DISC
!      print*,'MIRMAT: '
!      do NIT = 1, min(NBT,10)
!         print*,'MIRMAT: ',NIT,x1bt(NIT),y1bt(NIT),x2bt(NIT),y2bt(NIT)
!     enddo

      
      if (SHOW.eq.1) then
         print*,''
         write(*,'(''    2 - Add more stars to the lists'')')
      endif
      
      call incorp_buoy(x1bt,y1bt,x2bt,y2bt,NBT,
     .                 xin1,yin1,min1,NIN1,
     .                 xin2,yin2,min2,NIN2,
     .                 x1b,y1b,m1b,
     .                 x2b,y2b,m2b,NBUOY,DISC)
 
!      print*,'MIRMAT:  '
!      print*,'MIRMAT: exit incorp  12: '
!      print*,'MIRMAT:  '
!      print*,'MIRMAT:  NBUOY: ',NBUOY
!      print*,'MIRMAT: NITREJ: ',NITREJ
!      print*,'MIRMAT:  '
       
      if (SHOW.eq.1) then
         print*,''
         write(*,'(''    3 - Remove additional mismatches'')')
      endif
      
c----------------------------------------------
c
c this routine will remove the stray matches, allowing in
c the end only those that lie within the specified finding
c radius---as specified by darray(NITREJ)
c
!      print*,'MIRMAT:  '
!      print*,'MIRMAT: call purge_buoy: '
!      print*,'MIRMAT:  '
      do NIT = 1, NITREJ
         call purge_buoy(x1b,y1b,m1b,
     .                   x2b,y2b,m2b,NBUOY,NIT)
         enddo!NIT
!      print*,'MIRMAT:  '
!      print*,'MIRMAT: exit purge_buoy '
!      print*,'MIRMAT:  '
 
!      print*,'MIRMAT:  '
!      print*,'MIRMAT: NBUOY: ',NBUOY
!      print*,'MIRMAT:  '
 193  format(1x,a11,2f11.2)
      if (NBUOY.lt.10) then
         if (matmeth.eq.1) goto 112
         if (matmeth.eq.2) goto 113
         if (matmeth.eq.3) goto 114
         if (matmeth.eq.4) goto 115
         if (matmeth.eq.5) goto 116
         endif
!      print*,'MIRMAT:  '
!      print*,'MIRMAT:  '
!      write(*,193) 'MIRMAT:  xmin: ',minarr(x1b,NBUOY),minarr(x2b,NBUOY)
!      write(*,193) 'MIRMAT:  xmax: ',maxarr(x1b,NBUOY),maxarr(x2b,NBUOY)
!      write(*,193) 'MIRMAT:  ymin: ',minarr(y1b,NBUOY),minarr(y2b,NBUOY)
!      write(*,193) 'MIRMAT:  ymax: ',maxarr(y1b,NBUOY),maxarr(y2b,NBUOY)
!      write(*,193) 'MIRMAT:  mmin: ',minarr(m1b,NBUOY),minarr(m2b,NBUOY)
!      write(*,193) 'MIRMAT:  mmax: ',maxarr(m1b,NBUOY),maxarr(m2b,NBUOY)
!      print*,'MIRMAT:  '
!      print*,'MIRMAT:  '
!      print*,'MIRMAT: EXIT MIRMATM.F...'
!      print*,'MIRMAT:  '
!      print*,'MIRMAT:  '
!      print*,'MIRMAT:  '
 
      return
      end
 
 
      subroutine assume_orient(xin2,yin2,min2,NIN2,
     .                         x2b,y2b,m2b,NBs)
      implicit none
 
      integer NIN2
      real*8    xin2(NIN2), yin2(NIN2)
      real*8    min2(NIN2)
      real*8    x2b(*), y2b(*), m2b(*)
 
      integer NB, NBs
      integer  N2
 
      integer*2 array(401,401)
      integer i, j
      integer ii,jj
      real dx, dy, dd
      integer amax, imax, jmax, atot
 
      do i = 001, 401
      do j = 001, 401
         array(i,j) = 0
         enddo
         enddo
 
      do NB = 1, NBs
         do N2 = 1, NIN2
            dx = xin2(N2)-x2b(NB)
            dy = yin2(N2)-y2b(NB)
            i = 201 + dx
            j = 201 + dy
            if (i.gt.001.and.i.lt.401.and.
     .          j.gt.001.and.j.lt.401) then
                array(i  ,j  ) = array(i  ,j  ) + 1
                endif
            enddo
         enddo
 
      amax = 0
      imax = 0
      jmax = 0
      do i = 003, 399
      do j = 003, 399
         atot = 0
         do ii = i-2, i+2
         do jj = j-2, j+2
            dd = abs(ii-i) + abs(jj-j)
            atot = atot + array(ii,jj)*(1-dd/10.)
            enddo
            enddo
         if (atot.gt.amax) then
            amax = atot
            imax = i
            jmax = j
            endif
         enddo
         enddo
 
      dx = imax-201
      dy = jmax-201
 
!      print*,'    '
!      print*,'---> amax: ',amax
!      print*,'---> imax: ',imax,dx
!      print*,'---> jmax: ',jmax,dy
!      print*,'    '
 
      do NB = 1, NBs
         x2b(NB) = x2b(NB) + dx
         y2b(NB) = y2b(NB) + dy
         enddo
 
      return
      end
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/ROUTINES/MIRMAT/offschekFASTER.f"
c****                                         
c*********************************************
                                              
 
c
c this routine will take two lists of (x,y) positions
c and will determine what the offset between the lists
c is (assuming there is one); at this point it doesn't
c do anything reasonable if there ISN'T one, and I haven't
c made it robust enough to even know if it succeeded
c
 
 
      subroutine find_offset(x ,y , m, Ns ,
     .                       xx,yy,mm,NNs,OX,OY,
     .                       x1b,y1b,m1b,
     .                       x2b,y2b,m2b,NBUOY)
 
      implicit none
      real*8  x(*),  y(*),  m(*)
      integer  Ns
      real*8 xx(*), yy(*), mm(*)
      integer NNs
      real*8 OX, OY
      real*8 x1b(*),y1b(*),m1b(*)
      real*8 x2b(*),y2b(*),m2b(*)
      integer NBUOY
 
      integer  N
      integer NN
      integer  i,  j
      integer ii, jj
      integer offi, offj
      integer hmx, imx, jmx, hmx2
      integer hash(501,501)
      integer hu
      real*8 xsum, ysum
      character c(501)
      real*8 xmin, xmax
      real*8 ymin, ymax
      real*8    dx, dy, dd
      integer Nu, NNu
 
!      print*,'ENTER...   offschekFASTER... (slower)'
!      print*,'     ---   designed for large offsets, '
!      print*,'           and low precision...'
 
      Nu  = Ns
      NNu = NNs
 
      xmin = +9e9
      ymin = +9e9
      xmax = -9e9
      ymax = -9e9
      do NN = 1, NNu
         if (xx(NN).lt.xmin) xmin = xx(NN)
         if (xx(NN).gt.xmax) xmax = xx(NN)
         if (yy(NN).lt.ymin) ymin = yy(NN)
         if (yy(NN).gt.ymax) ymax = yy(NN)
         enddo
 
!      print*,'Nu       : ',Nu
!      print*,'NNu      : ',NNu
!      print*,'NN:  xmin: ',xmin
!      print*,'NN:  xmax: ',xmax
!      print*,'NN:  ymin: ',ymin
!      print*,'NN:  ymax: ',ymax
!      print*,'           '
 
      do ii = 1, 501
      do jj = 1, 501
         hash(ii,jj) = 0
         enddo
         enddo
      do N  = 1, Ns
         do NN = 1, NNs
            offi = (xx(NN)-x(N))/2 + 0.5
            offj = (yy(NN)-y(N))/2 + 0.5
            if (abs(offi).le.250.and.
     .          abs(offj).le.250) then
                  hash(251+offi,251+offj) =
     .            hash(251+offi,251+offj) + 1
                  endif
            enddo
 66      continue
         enddo
 
      hmx = 0
      imx = 0
      jmx = 0
      do i = 1, 500
      do j = 1, 500
         hu = hash(i,j)+hash(i+1,j)+hash(i,j+1)+hash(i+1,j+1)
         if (hu.gt.hmx) then
            hmx = hu
            imx = i
            jmx = j
            endif
         enddo
         enddo
 
      hmx2 = 0
      do i = 1, 100
      do j = 1, 100
         hu = hash(i,j)+hash(i+1,j)+hash(i,j+1)+hash(i+1,j+1)
         if (hu.gt.hmx2.and.
     .       abs(imx-i).gt.5.and.
     .       abs(jmx-j).gt.5) then
            hmx2 = hu
            endif
         enddo
         enddo
 
 
 
!      write(*,144) (i-251,' ',i=251-15,251+15)
!      do j = 266,236, -1
!         do i = 1, 501
!            c(i) = ' '
!            if (i.eq.251) c(i) = '*'
!            if (j.eq.251) c(i) = '*'
!            enddo
!         write(*,145) j-251,(hash(i,j),c(i),
!     .                i=251-15,251+15)
!         enddo
 
!      print*,'IMX: ',imx
!      print*,'JMX: ',jmx
!      print*,'HMX: ',hmx,hash(imx,jmx)
!      print*,'HMx: ',hmx2
!      print*,' Nu: ',NU
!      print*,'ABC: '
 
c      if (hmx.lt.2*hmx2) then
c         print*,' ---> NO SIGNIF MATCHES...'
c         NBUOY = 0
c         return
c         endif
 
!      write(*,144) (i-251,' ',i=max(imx-10,1),min(imx+10,501))
!      do j = min(jmx+10,501), max(jmx-10,001), -1
!         do i = 1, 251
!            c(i) = ' '
!            if (i.eq.51) c(i) = '*'
!            if (j.eq.51) c(i) = '*'
!            enddo
!         write(*,145) j-251,(hash(i,j),c(i),
!     .                i=max(imx-10,1),min(imx+10,501))
!         enddo
 144  format(1x,4x,1x,501(i5,a1))
 145  format(1x,i4,1x,501(i5,a1))
 
 
      OX = imx-251 + 0.5
      OY = jmx-251 + 0.5
!      print*,'Nu: ',NU
!      print*,'OX: ',OX
!      print*,'OY: ',OY
 
      OX = OX * 2
      OY = OY * 2
!      print*,'ADJ-----'
!      print*,'OX: ',OX
!      print*,'OY: ',OY
 
      xsum = 0.
      ysum = 0.
      NBUOY = 0.
!      print*,' '
!      print*,'  Nu: ',Nu
!      print*,' NNu: ',NNu
!      print*,' '
      NBUOY = 0
      do N = 1, Nu
         do NN  = 1, NNu
            dx = x(N)-xx(NN)+OX
            dy = y(N)-yy(NN)+OY
            dd = sqrt(dx**2+dy**2)
            if (dd.lt.5.0) then
               NBUOY = NBUOY + 1
               if (NBUOY.gt._NMATMX_) then
                  print*,'       N : ',N
                  print*,'       NU: ',NU
                  print*,'    NBUOY: ',NBUOY
                  print*,' _NMATMX_: ',_NMATMX_
                  stop
                  endif
               x1b(NBUOY) =  x( N)
               y1b(NBUOY) =  y( N)
               m1b(NBUOY) =  m( N)
               x2b(NBUOY) = xx(NN)
               y2b(NBUOY) = yy(NN)
               m2b(NBUOY) = mm(NN)
               xsum = xsum + x(N)-xx(NN)
               ysum = ysum + y(N)-yy(NN)
               endif
            enddo
         enddo
 
!      print*,'  NEW----'
!      print*,'NBUOY: ',NBUOY
      OX = xsum/NBUOY
      OY = ysum/NBUOY
!      print*,'  OX: ',OX
!      print*,'  OY: ',OY
 
      return
      end
 
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/ROUTINES/MIRMAT/MM50CHEK.F"
c****                                         
c*********************************************
                                              
c
c a new routine, which will just stick with the
c brightest 50 and will find matches among them
c using the V/VMAX technique...
c
 
 
c--------------------------------------------------------------------
c
c this subroutine will take TWO photometry/astrometry lists and will
c (1) determine if there is any overlap between the two fields, and
c (2) if there is, it will give a list of stars that are the same
c     in the two images (which will allow for a local transformation)
c
c--------------------------------------------------------------------
 
      subroutine miracle_match50(xin1,yin1,min1,NIN1,
     .                           xin2,yin2,min2,NIN2,
     .                           x1b,y1b,m1b,
     .                           x2b,y2b,m2b,NBUOY)
      implicit none
 
      integer NIN1
      real*8    xin1(NIN1), yin1(NIN1), min1(NIN1)
      integer NIN2
      real*8    xin2(NIN2), yin2(NIN2), min2(NIN2)
      real*8    x1b(*), y1b(*), x2b(*), y2b(*)
      real*8    m1b(*), m2b(*)
      integer NBUOY
 
      real*8 x1(_NSTMAX_), y1(_NSTMAX_), m1(_NSTMAX_)
      real*8 x2(_NSTMAX_), y2(_NSTMAX_), m2(_NSTMAX_)
 
      real*8 atan2dj
 
      real*8 dab, dac, dmax, dmin, vvmax
      real*8 vdprod, vcprod, angl
      integer iv, ia
      integer na, nb, nc
      real*8    xa, xb, xc
      real*8    ya, yb, yc
      real*8    ma, mb!,mc
      byte nau(201,360)
      byte nbu(201,360)
      byte ncu(201,360)
      integer notel(50)
      integer votel(50)
      integer voteb(50,50)
      integer nmat(50)
 
      integer NIN1u, NIN2u
 
 
      integer nmiss
      integer ndone
 
 
      print*,' '
      print*,'  MM50:  mirmat50:  use brightest 50'
      print*,'  MM50:  '
      print*,'  MM50:  '
 
      if (NIN1.lt.50.or.NIN2.lt.50) then
         print*,'You need at least 50 to '
         print*,'find the matches...'
         print*,'NIN1: ',NIN1
         print*,'NIN2: ',NIN2
         stop
         endif
 
      NIN1u = 50
      NIN2u = 50
      call ord_brite(xin1,yin1,min1,NIN1,x1,y1,m1,NIN1u,50)
      print*,'  MM50: OB1: ',NIN1,NIN1u
      call ord_brite(xin2,yin2,min2,NIN2,x2,y2,m2,NIN2u,50)
      print*,'  MM50: OB2: ',NIN2,NIN2u
 
      print*,'  MM50: '
      print*,'  MM50: ORD_BRITE: '
      print*,'  MM50: '
c      do n = 1, 50
c         write(*,150) n, x1(n),y1(n),m1(n),x2(n),y2(n),m2(n)
c 150     format(1x,i2,1x,2(f10.4,1x,f10.4,1x,f7.3,3x))
c         enddo
 
      print*,'  MM50: '
      print*,'  MM50: DO V/VMAX-type search...'
      print*,'  MM50: '
      do na = 1, 48
         xa = x1(na)
         ya = y1(na)
         do nb = na+1, 49
         do nc = nb+1, 50
            xb = x1(nb)
            yb = y1(nb)
            xc = x1(nc)
            yc = y1(nc)
            if (xa.eq.xb.and.ya.eq.yb) goto 5
            if (xa.eq.xc.and.ya.eq.yc) goto 5
            if (xb.eq.xc.and.yb.eq.yc) goto 5
            dab = sqrt((xa-xb)**2+(ya-yb)**2)
            dac = sqrt((xa-xc)**2+(ya-yc)**2)
            dmax = max(dab,dac)
            dmin = min(dab,dac)
            vvmax = dmin**2/dmax**2
            if (dab.lt.dac) vvmax = -vvmax
            vdprod = (xb-xa)*(xc-xa) + (yb-ya)*(yc-ya)
            vcprod = (xb-xa)*(yc-ya) - (yb-ya)*(xc-xa)
            angl = atan2dj(vdprod,vcprod)
            if (angl.lt.  0) angl = angl+360.
            if (angl.gt.360) angl = angl-360.
            iv = 101+100*vvmax
            ia = max(angl,1.0)
            if (nau(iv,ia).ne.0) nmiss = nmiss + 1
            if (nau(iv,ia).eq.0) then
               ndone = ndone + 1
               nau(iv,ia) = na
               nbu(iv,ia) = nb
               ncu(iv,ia) = nc
               endif
 5          continue
            enddo
            enddo
         enddo
 
c     write(*,114)
c 114  format('N2',2x,'N1',3x,'..x2...',1x,'..y2...',1x,'.m2..',
c     .                    3x,'..x1...',1x,'..y1...',1x,'.m1..',
c     .                    3x,' MAX NXT NX2 NX3 ......')
      do nb = 1, 50
         do nc = 1, 50
            voteb(nb,nc) = 0
            enddo
         enddo
      do na = 1, 48
         nmat(na) = 0
         do nb = 1, 50
            notel(nb) = nb
            votel(nb) = 0
            enddo
         xa = x2(na)
         ya = y2(na)
         do nb = na+1, 49
         do nc = nb+1, 50
            xb = x2(nb)
            yb = y2(nb)
            xc = x2(nc)
            yc = y2(nc)
            if (xa.eq.xb.and.ya.eq.yb) goto 6
            if (xa.eq.xc.and.ya.eq.yc) goto 6
            if (xb.eq.xc.and.yb.eq.yc) goto 6
            dab = sqrt((xa-xb)**2+(ya-yb)**2)
            dac = sqrt((xa-xc)**2+(ya-yc)**2)
            dmax = max(dab,dac)
            dmin = min(dab,dac)
            vvmax = dmin**2/dmax**2
            if (dab.lt.dac) vvmax = -vvmax
            vdprod = (xb-xa)*(xc-xa) + (yb-ya)*(yc-ya)
            vcprod = (xb-xa)*(yc-ya) - (yb-ya)*(xc-xa)
            angl = atan2dj(vdprod,vcprod)
            if (angl.lt.  0) angl = angl+360.
            if (angl.gt.360) angl = angl-360.
            iv = 101+100*vvmax
            ia = max(angl,1.)
            if (nau(iv,ia).ne.0) then
               voteb(nau(iv,ia),na) = voteb(nau(iv,ia),na) + 1
               voteb(nbu(iv,ia),nb) = voteb(nbu(iv,ia),nb) + 1
               voteb(ncu(iv,ia),nc) = voteb(ncu(iv,ia),nc) + 1
               endif
  6         continue
            enddo
            enddo
         do nb = 1, 50
            votel(nb) = voteb(nb,na)
            enddo
         call rcosort2ii(votel,notel,50,10)
         if (votel(1).gt.2*votel(2)) nmat(na) = notel(1)
         xa = x2(na)
         ya = y2(na)
         ma = m2(na)
         xb = 0.
         yb = 0.
         mb = 0.
         nb = nmat(na)
         if (nb.ne.0) then
            xb = x1(nb)
            yb = y1(nb)
            mb = m1(nb)
            endif
c         write(*,115) na,nb,xa,ya,ma,
c     .                      xb,yb,mb,(votel(nb),nb=1,10)
c 115     format(i2,2x,i2,3x,f7.2,1x,f7.2,1x,f5.1,3x,
c     .                      f7.2,1x,f7.2,1x,f5.1,3x,10i4)
         enddo
 
 
c
c if we were able to find any matches, then
c log them
c
      NBUOY = 0
      do na = 1, 48
         if (nmat(na).ne.0) then
            NBUOY = NBUOY + 1
            nb = nmat(na)
            x1b(NBUOY) = x1(nb)
            y1b(NBUOY) = y1(nb)
            m1b(NBUOY) = m1(nb)
            x2b(NBUOY) = x2(na)
            y2b(NBUOY) = y2(na)
            m2b(NBUOY) = m2(na)
 116        format(i4,1x,i2,1x,i2,1x,6f8.2)
            endif
         enddo
 
 97   continue
 
      print*,'  MM50: '
      print*,'  MM50: found NBUOY: ',NBUOY
      print*,'  MM50: '
      return
 
      end
 
 
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/ROUTINES/MIRMAT/MM25CHEK.F"
c****                                         
c*********************************************
                                              
c
c a new routine, which will just stick with the
c brightest 25 and will find matches among them
c using the V/VMAX technique...
c
 
 
c--------------------------------------------------------------------
c
c this subroutine will take TWO photometry/astrometry lists and will
c (1) determine if there is any overlap between the two fields, and
c (2) if there is, it will give a list of stars that are the same
c     in the two images (which will allow for a local transformation)
c
c--------------------------------------------------------------------
 
      subroutine miracle_match25(xin1,yin1,min1,NIN1,
     .                           xin2,yin2,min2,NIN2,
     .                           x1b,y1b,m1b,
     .                           x2b,y2b,m2b,NBUOY)
      implicit none
 
      integer NIN1
      real*8  xin1(*), yin1(*), min1(*)
      integer NIN2
      real*8  xin2(*), yin2(*), min2(*)
      real*8  x1b(*), y1b(*), x2b(*), y2b(*)
      real*8  m1b(*), m2b(*)
      integer NBUOY
 
      real*8 x1(_NMATMX_), y1(_NMATMX_), m1(_NMATMX_)
      real*8 x2(_NMATMX_), y2(_NMATMX_), m2(_NMATMX_)
 
      integer   n
 
      real*8 atan2dj
      real*8 dab, dac, dmax, dmin, vvmax
      real*8 vdprod, vcprod, angl
      integer iang
      integer iv, ia
      integer na, nb, nc
      real*8    xa, xb, xc
      real*8    ya, yb, yc
      real*8    ma, mb!,mc
 
      integer nau(201,360)
      integer nbu(201,360)
      integer ncu(201,360)
      integer notel(50)
      integer votel(50)
      integer voteb(50,50)
      integer nmat(50)
 
      integer NIN1u, NIN2u
 
 
      integer nmiss
      integer ndone
 
      nmiss = 0
      ndone = 0
 
!      print*,'  MM25:  '
!      print*,'  MM25: mirmat25:  use brightest 25'
!      print*,'  MM25:  '
 
      if (NIN1.lt.25.or.NIN2.lt.25) then
         NBUOY = 0
!         print*,'  MM25:  You need at least 25 to '
!         print*,'  MM25:  find the matches with MM25'
!         print*,'  MM25:   NIN1: ',NIN1
!         print*,'  MM25:   NIN2: ',NIN2
!         print*,'  MM25:  NBUOY: ',NBUOY
!         print*,'  MM25: '
         return
         endif
 
      NIN1u = 25
      NIN2u = 25
      call ord_brite(xin1,yin1,min1,NIN1,x1,y1,m1,NIN1u,25)
!      print*,'  MM25: OB1: ',NIN1,NIN1u
      call ord_brite(xin2,yin2,min2,NIN2,x2,y2,m2,NIN2u,25)
!      print*,'  MM25: OB2: ',NIN2,NIN2u
 
!      print*,'  MM25: '
!      print*,'  MM25: ORD_BRITE: '
!      print*,'  MM25: '
!      do n = 1, 25
!         write(*,150) n, x1(n),y1(n),m1(n),x2(n),y2(n),m2(n)
 150     format(1x,i2,1x,2(f10.4,1x,f10.4,1x,f7.3,3x))
!         enddo
 
      do iv = 001, 201
      do ia = 001, 360
         nau(iv,ia) = 0
         nbu(iv,ia) = 0
         ncu(iv,ia) = 0
         enddo
         enddo
 
      do ia = 1, 50
         notel(ia) = 0
         votel(ia) = 0
          nmat(ia) = 0
         do iv = 1, 50
            voteb(ia,iv) = 0
            enddo
         enddo
 
 
!      print*,'  MM25:  do V/VMAX test...'
      do na = 1, 23
         xa = x1(na)
         ya = y1(na)
         do nb = na+1, 24
         do nc = nb+1, 25
            xb = x1(nb)
            yb = y1(nb)
            xc = x1(nc)
            yc = y1(nc)
            if (xa.eq.xb.and.ya.eq.yb) goto 5
            if (xa.eq.xc.and.ya.eq.yc) goto 5
            if (xb.eq.xc.and.yb.eq.yc) goto 5
            dab = sqrt((xa-xb)**2+(ya-yb)**2)
            dac = sqrt((xa-xc)**2+(ya-yc)**2)
            dmax = max(dab,dac)
            dmin = min(dab,dac)
            vvmax = dmin**2/dmax**2
            if (dab.lt.dac) vvmax = -vvmax
            vdprod = (xb-xa)*(xc-xa) + (yb-ya)*(yc-ya)
            vcprod = (xb-xa)*(yc-ya) - (yb-ya)*(xc-xa)
            angl = atan2dj(vdprod,vcprod)
            if (angl.lt.  0) angl = angl+360.
            if (angl.gt.360) angl = angl-360.
            iv = 101+100*vvmax
            ia = max(angl,1.0)
            if (nau(iv,ia).ne.0) nmiss = nmiss + 1
            if (nau(iv,ia).eq.0) then
               ndone = ndone + 1
               nau(iv,ia) = na
               nbu(iv,ia) = nb
               ncu(iv,ia) = nc
               endif
 5          continue
            enddo
            enddo
         enddo
 
c      write(*,114)
c 114  format('N2',2x,'N1',3x,'..x2...',1x,'..y2...',1x,'.m2..',
c     .                    3x,'..x1...',1x,'..y1...',1x,'.m1..',
c     .                    3x,' MAX NXT NX2 NX3 ......')
      do nb = 1, 25
         do nc = 1, 25
            voteb(nb,nc) = 0
            enddo
         enddo
      do na = 1, 23
         nmat(na) = 0
         do nb = 1, 25
            notel(nb) = nb
            votel(nb) = 0
            enddo
         xa = x2(na)
         ya = y2(na)
         do nb = na+1, 24
         do nc = nb+1, 25
            xb = x2(nb)
            yb = y2(nb)
            xc = x2(nc)
            yc = y2(nc)
            if (xa.eq.xb.and.ya.eq.yb) goto 6
            if (xa.eq.xc.and.ya.eq.yc) goto 6
            if (xb.eq.xc.and.yb.eq.yc) goto 6
            dab = sqrt((xa-xb)**2+(ya-yb)**2)
            dac = sqrt((xa-xc)**2+(ya-yc)**2)
            dmax = max(dab,dac)
            dmin = min(dab,dac)
            vvmax = dmin**2/dmax**2
            if (dab.lt.dac) vvmax = -vvmax
            vdprod = (xb-xa)*(xc-xa) + (yb-ya)*(yc-ya)
            vcprod = (xb-xa)*(yc-ya) - (yb-ya)*(xc-xa)
            angl = atan2dj(vdprod,vcprod)
            if (angl.lt.  0) angl = angl+360.
            if (angl.gt.360) angl = angl-360.
            iv = 101+100*vvmax
            ia = max(angl,1.0)
            if (nau(iv,ia).ne.0) then
               voteb(nau(iv,ia),na) = voteb(nau(iv,ia),na) + 1
               voteb(nbu(iv,ia),nb) = voteb(nbu(iv,ia),nb) + 1
               voteb(ncu(iv,ia),nc) = voteb(ncu(iv,ia),nc) + 1
               endif
  6         continue
            enddo
            enddo
 
         do nb = 1, 25
            votel(nb) = voteb(nb,na)
            enddo
         call rcosort2ii(votel,notel,25,10)
         if (votel(1).gt.2*votel(2)) nmat(na) = notel(1)
         xa = x2(na)
         ya = y2(na)
         ma = m2(na)
         xb = 0.
         yb = 0.
         mb = 0.
         nb = nmat(na)
         if (nb.ne.0) then
            xb = x1(nb)
            yb = y1(nb)
            mb = m1(nb)
            endif
c         write(*,115) na,nb,xa,ya,ma,
c     .                      xb,yb,mb,(votel(nb),nb=1,10)
c 115     format(i2,2x,i2,3x,f7.2,1x,f7.2,1x,f5.1,3x,
c     .                      f7.2,1x,f7.2,1x,f5.1,3x,10i4)
         enddo
 
 
      NBUOY = 0
      do na = 1, 48
         if (nmat(na).ne.0) then
            NBUOY = NBUOY + 1
            nb = nmat(na)
            x1b(NBUOY) = x1(nb)
            y1b(NBUOY) = y1(nb)
            m1b(NBUOY) = m1(nb)
            x2b(NBUOY) = x2(na)
            y2b(NBUOY) = y2(na)
            m2b(NBUOY) = m2(na)
 116        format(i4,1x,i2,1x,i2,1x,6f8.2)
            endif
         enddo
 
 97   continue
 
!      print*,'  MM25: '
!      print*,'  MM25: found NBUOY: ',NBUOY
!      print*,'  MM25: '
      return
 
      end
 
 
 
 
 
 
 
c      program mirmat_sns
c      implicit none
c
c      character*80 STRING
c
c      integer I1, I1s
c      real*8 x1i(99), y1i(99), m1i(99)
c      integer I2, I2s
c      real*8 x2i(99), y2i(99), m2i(99)
c
c      open(10,file='LIST1',status='old')
c      I1s = 0
c   11 continue
c      read(10,'(a80)',end=12) STRING
c      if (STRING(1:1).eq.'#') goto 11
c      read(STRING,*) x1i(I1s+1), y1i(I1s+1), m1i(I1s+1)
c      I1s = I1s+1
c      goto 11
c   12 close(10)
c      print*,'---> I1s: ',I1s
c
c      open(20,file='LIST2',status='old')
c      I2s = 0
c   21 continue
c      read(20,'(a80)',end=22) STRING
c      if (STRING(1:1).eq.'#') goto 21
c      read(STRING,*) x2i(I2s+1), y2i(I2s+1), m2i(I2s+1)
c      I2s = I2s+1
c      goto 21
c   22 close(20)
c      print*,'---> I2s: ',I2s
c
c
c      call find_match_sns(x1i,y1i,m1i,I1s,
c     .                    x2i,y2i,m2i,I2s)
c
c      stop
c      end
c
 
 
 
      subroutine find_match_sns(xin1,yin1,min1,NIN1,
     .                          xin2,yin2,min2,NIN2,
     .                          x1b,y1b,m1b,
     .                          x2b,y2b,m2b,NBUOY)
      implicit none
 
      integer NIN1
      real*8  xin1(*), yin1(*), min1(*)
      integer NIN2
      real*8  xin2(*), yin2(*), min2(*)
      real*8  x1b(*), y1b(*), x2b(*), y2b(*)
      real*8  m1b(*), m2b(*)
      integer NBUOY
 
 
      real*8 AG,BG,CG,DG
      real*8 Xo,Yo,Wo,Zo
 
      integer N1
      integer N2
 
      integer I1, I1s
      real*8 x1i(999), y1i(999), m1i(999)
      integer I2, I2s
      real*8 x2i(999), y2i(999), m2i(999)
 
      integer I1a, I1b, I1c
      integer I2a, I2b, I2c
 
      real*8 dx1, dy1, dd1
      real*8 dx2, dy2, dd2
 
      real*8 fx, fy
 
      real*8  ddx1, ddy1, ddd1, ddx1e, ddy1e
      real*8  ddx2, ddy2, ddd2, ddx2e, ddy2e
 
      integer N
 
      real*8  x2e, y2e
      integer T, Ts, Tmax
      real*8 x1t(25), y1t(25), m1t(25)
      real*8 x2t(25), y2t(25), m2t(25)
 
      if (NIN1.gt.9999) stop 'NIN1.gt.9999'
      if (NIN2.gt.9999) stop 'NIN2.gt.9999'
 
      NBUOY = 0
 
 
      if (NIN1.gt.999) stop 'FIND_MATCH_SNS NIN1.gt.999'
      do N1 = 1, NIN1
         x1i(N1) = xin1(N1)
         y1i(N1) = yin1(N1)
         m1i(N1) = min1(N1)
         enddo
 
      if (NIN2.gt.999) stop 'FIND_MATCH_SNS NIN2.gt.999'
      do N2 = 1, NIN2
         x2i(N2) = xin2(N2)
         y2i(N2) = yin2(N2)
         m2i(N2) = min2(N2)
         enddo
 
      call cosort3d(m1i,x1i,y1i,NIN1)
      call cosort3d(m2i,x2i,y2i,NIN2)
 
      I1s = min(NIN1,25)
      I2s = min(NIN2,25)
 
      Tmax = 0
      do I1a = 1, I1s
      do I1b = 1, I1s
      do I2a = 1, I2s
      do I2b = 1, I2s
         if (I1a.ne.I1b.and.I2a.ne.I2b) then
             Ts = 0
             dx1 = x1i(I1b)-x1i(I1a)
             dy1 = y1i(I1b)-y1i(I1a)
             dd1 = sqrt(dx1**2+dy1**2)
             dx2 = x2i(I2b)-x2i(I2a)
             dy2 = y2i(I2b)-y2i(I2a)
             dd2 = sqrt(dx2**2+dy2**2)
             do I1c = 1, I1s
                ddx1 = x1i(I1c)-x1i(I1a)
                ddy1 = y1i(I1c)-y1i(I1a)
                ddd1 = sqrt(ddx1**2+ddy1**2)
                fx = ( dx1*ddx1 + dy1*ddy1)/dd1/dd1
                fy = (-dx1*ddy1 + dy1*ddx1)/dd1/dd1
                x2e =  x2i(I2a) + fx*dx2 + fy*dy2
                y2e =  y2i(I2a) + fx*dy2 - fy*dx2
c               write(*,119) I1a,I1b,I2a,I2b,I1c,
c    .                       x1i(I1c),y1i(I1c),
c    .                            x2e-x2i(I2a),     y2e-y2i(I2a),
c    .                       x2i(I1c)-x2i(I2a),y2i(I1c)-y2i(I2a)
c 119           format(1x,4i3,5x,i3,1x,8f10.2)
                do I2c = 1, I2s
                   if ((x2i(I2c)-x2e)**2
     .                +(y2i(I2c)-y2e)**2.le.5**2) then
                      Ts = Ts + 1
                      x1t(Ts) = x1i(I1c)
                      y1t(Ts) = y1i(I1c)
                      m1t(Ts) = m1i(I1c)
                      x2t(Ts) = x2i(I2c)
                      y2t(Ts) = y2i(I2c)
                      m2t(Ts) = m2i(I2c)
                      endif
                   enddo
                enddo
             if (Ts.gt.Tmax) then
                Tmax = Ts
                write(*,113) I1a,I1b,I2a,I2b,Ts,dd2/dd1
  113           format(1x,2i3,3x,2i3,3x,i5,1x,f12.7)
                NBUOY = Ts
                do T = 1, Ts
                   x1b(T) =  x1t(T)
                   y1b(T) =  y1t(T)
                   m1b(T) =  m1t(T)
                   x2b(T) =  x2t(T)
                   y2b(T) =  y2t(T)
                   m2b(T) =  m2t(T)
                   enddo
                endif
             endif
         enddo
         enddo
         enddo
         enddo
 
      do T = 1, NBUOY
         write(*,'(i4,1x,3f10.3,5x,3f10.3)')
     .              T,x1b(T),y1b(T),m1b(T),
     .                x2b(T),y2b(T),m2b(T)
         enddo
 
      return
      end
 
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/ROUTINES/MIRMAT/MM3DCHEK.F"
c****                                         
c*********************************************
                                              
 
c-------------------------------------------------------
c
c this will try to match two star lists by taking the brightest
c 50 stars in each list and trying every permutation among them
c to see if we can find a combination of them that works
c
      subroutine miracle_match3d(xin1,yin1,min1,NIN1,
     .                           xin2,yin2,min2,NIN2,
     .                           x1b,y1b,m1b,
     .                           x2b,y2b,m2b,NBUOY)
      implicit none
 
      integer NIN1
      real*8    xin1(NIN1), yin1(NIN1), min1(NIN1)
      integer NIN2
      real*8    xin2(NIN2), yin2(NIN2), min2(NIN2)
      real*8    x1b(999), y1b(999)
      real*8    x2b(999), y2b(999)
      real*8    m1b(999), m2b(999)
      integer NBUOY
 
      real*8 x1(_NSTMAX_), y1(_NSTMAX_), m1(_NSTMAX_)
      real*8 x2(_NSTMAX_), y2(_NSTMAX_), m2(_NSTMAX_)
 
      integer n1, n2
 
      real*8 AG, BG, CG, DG
      real*8 x1o, y1o
      real*8 x2o, y2o
      real*8 x2t, y2t
      integer nmu
      real*8    dmu
 
      integer na, nb, nc
      real*8  xa, xb, xc
      real*8  ya, yb, yc
 
      integer NIN1u, NIN2u
 
 
      real*8  dll(3)
      integer nll(3)
 
      integer nmin
      real*8    dmin, dd
      integer NL, NLs
      real*8    d1l(99999), d2l(99999), d3l(99999)
      integer n1l(99999), n2l(99999), n3l(99999)
 
      NBUOY = 0
 
      if (NIN1.lt.50.or.NIN2.lt.50) then
!         print*,'You need at least 50 to '
!         print*,'find the matches...'
!         print*,'NIN1: ',NIN1
!         print*,'NIN2: ',NIN2
         return
         endif
 
      NIN1u = 50
      NIN2u = 50
      call ord_brite(xin1,yin1,min1,NIN1,x1,y1,m1,NIN1u,50)
!      print*,'MM3D OB1: ',NIN1,NIN1u
      call ord_brite(xin2,yin2,min2,NIN2,x2,y2,m2,NIN2u,50)
!      print*,'MM3D OB2: ',NIN2,NIN2u
 
!      print*,'MM3D:  '
!      print*,'MM3D: ORD_BRITE: '
!      print*,'MM3D:  '
c      do n = 1, 50
c         if ((n.le.010).or.
c     .       (n.ge.010.and.n.le.100.and.n.eq.n/10*10).or.
c     .       (n.gt.100.and.n.eq.n/100*100))
c     .   write(*,150) n, x1(n),y1(n),m1(n),x2(n),y2(n),m2(n)
c 150     format(1x,i3,1x,2(f10.4,1x,f10.4,1x,f7.3,3x))
c         enddo
 
      NLs = 0
!      print*,'MM3D:  '
!      print*,'MM3D: TRY PERMUTATIONS, VOTE FOR GOOD ONES'
!      print*,'MM3D:  '
      do na = 1, 50-2
         xa = x1(na)
         ya = y1(na)
         do nb = na+1, 50-1
         do nc = nb+1, 50-0
            xb = x1(nb)
            yb = y1(nb)
            xc = x1(nc)
            yc = y1(nc)
            if (xa.eq.xb.and.ya.eq.yb) goto 5
            if (xa.eq.xc.and.ya.eq.yc) goto 5
            if (xb.eq.xc.and.yb.eq.yc) goto 5
            dll(1) = sqrt((xa-xb)**2+(ya-yb)**2)
            dll(2) = sqrt((xa-xc)**2+(ya-yc)**2)
            dll(3) = sqrt((xb-xc)**2+(yb-yc)**2)
            if (dll(1).gt.2500) goto 5
            if (dll(2).gt.2500) goto 5
            if (dll(3).gt.2500) goto 5
            nll(1) = nc
            nll(2) = nb
            nll(3) = na
            call cosort2di(dll,nll,3)
            NLs = NLs + 1
            if (NLs.gt.99999) then
               print*,'MM3DCHECK: NLs EXCESS: ',NLs,na,nb,nc
               stop
               endif
            d1l(NLs) = dll(1)
            d2l(NLs) = dll(2)
            d3l(NLs) = dll(3)
            n1l(NLs) = nll(1)
            n2l(NLs) = nll(2)
            n3l(NLs) = nll(3)
 5          continue
            enddo
            enddo
         enddo
 
!      print*,'MM3D:  '
!      print*,'MM3D: TALLY UP THE GOOD VOTES...'
!      print*,'MM3D:  '
      do na = 1, 10
         xa = x2(na)
         ya = y2(na)
         do nb = na+1, 50-1
         do nc = nb+1, 50
            xb = x2(nb)
            yb = y2(nb)
            xc = x2(nc)
            yc = y2(nc)
            dll(1) = sqrt((xa-xb)**2+(ya-yb)**2)
            dll(2) = sqrt((xa-xc)**2+(ya-yc)**2)
            dll(3) = sqrt((xb-xc)**2+(yb-yc)**2)
            if (dll(1).gt.2500) goto 7
            if (dll(2).gt.2500) goto 7
            if (dll(3).gt.2500) goto 7
            nll(1) = nc
            nll(2) = nb
            nll(3) = na
            call cosort2di(dll,nll,3)
            dmin =   2.0
            nmin =   0.0
            do NL = 1, NLs
               dd = sqrt((dll(1)-d1l(NL))**2+
     .                   (dll(2)-d2l(NL))**2+
     .                   (dll(3)-d3l(NL))**2)
               if (dd.lt.dmin) then
                  dmin = dd
                  nmin = NL
                  endif
               enddo
            if (nmin.ne.0) then
               NBUOY = 3
               x1b(1) = x1(n1l(nmin))
               y1b(1) = y1(n1l(nmin))
               m1b(1) = m1(n1l(nmin))
               x1b(2) = x1(n2l(nmin))
               y1b(2) = y1(n2l(nmin))
               m1b(2) = m1(n2l(nmin))
               x1b(3) = x1(n3l(nmin))
               y1b(3) = y1(n3l(nmin))
               m1b(3) = m1(n3l(nmin))
               x2b(1) = x2(nll(1))
               y2b(1) = y2(nll(1))
               m2b(1) = m2(nll(1))
               x2b(2) = x2(nll(2))
               y2b(2) = y2(nll(2))
               m2b(2) = m2(nll(2))
               x2b(3) = x2(nll(3))
               y2b(3) = y2(nll(3))
               m2b(3) = m2(nll(3))
               call glob_fit6nrDP(x1b,y1b,x2b,y2b,3,
     .                            AG,BG,CG,DG,x1o,y1o,x2o,y2o)
               NBUOY = 0
               do n1 = 1, 50
                  x2t = AG*(x1(n1)-x1o)+BG*(y1(n1)-y1o)+x2o
                  y2t = CG*(x1(n1)-x1o)+DG*(y1(n1)-y1o)+y2o
                  dmu = 2.5
                  nmu = 0
                  do n2 = 1, 50
                     dd = sqrt((x2(n2)-x2t)**2+(y2(n2)-y2t)**2)
                     if (dd.lt.dmu) then
                        dmu = dd
                        nmu = n2
                        endif
                     enddo
                  if (nmu.ne.0) then
                     NBUOY = NBUOY + 1
                     x1b(NBUOY) = x1(n1)
                     y1b(NBUOY) = y1(n1)
                     m1b(NBUOY) = m1(n1)
                     x2b(NBUOY) = x2(nmu)
                     y2b(NBUOY) = y2(nmu)
                     m2b(NBUOY) = m2(nmu)
                     endif
                  enddo
!               write( *,147) na, nb, nc, nmin, dmin,
!     .                       dll(1),dll(2),dll(3),NBUOY
 147           format(1x,4i7,1x,f4.1,1x,3f10.2,1x,i3)
               if (NBUOY.ge.10) then
                  call glob_fit6nrDP(x1b,y1b,x2b,y2b,NBUOY,
     .                               AG,BG,CG,DG,x1o,y1o,x2o,y2o)
                  AG = (AG+DG)/2
                  BG = (BG-CG)/2
                  DG =  AG
                  CG = -BG
                  NBUOY = 0
                  do n1 = 1, nin1
                     x2t = AG*(xin1(n1)-x1o)+BG*(yin1(n1)-y1o)+x2o
                     y2t = CG*(xin1(n1)-x1o)+DG*(yin1(n1)-y1o)+y2o
                     dmu = 1.5
                     nmu = 0
                     do n2 = 1, nin2
                        dd = sqrt((xin2(n2)-x2t)**2+(yin2(n2)-y2t)**2)
                        if (dd.lt.dmu) then
                           dmu = dd
                           nmu = n2
                           endif
                        enddo
                     if (nmu.ne.0.and.min2(nmu).le.-3.5) then
                        NBUOY = NBUOY + 1
                        x1b(NBUOY) = xin1(n1)
                        y1b(NBUOY) = yin1(n1)
                        m1b(NBUOY) = min1(n1)
                        x2b(NBUOY) = xin2(nmu)
                        y2b(NBUOY) = yin2(nmu)
                        m2b(NBUOY) = min2(nmu)
                        endif
                     enddo
                  return
                  endif
               endif
   7        continue
            enddo
            enddo
         enddo
 
      NBUOY = 0
      return
 
      end
 
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/ROUTINES/MIRMAT/MM99CHEK.F"
c****                                         
c*********************************************
                                              
 
c------------------------------------------------------------
c
c do a local-based search for similar triangles
c
      subroutine miracle_match99(xin1,yin1,min1,NIN1,
     .                           xin2,yin2,min2,NIN2,
     .                           x1b,y1b,m1b,x2b,y2b,m2b,NBUOY)
      implicit none
 
      integer NIN1
      real*8    xin1(NIN1), yin1(NIN1), min1(NIN1)
      integer NIN2
      real*8    xin2(NIN2), yin2(NIN2), min2(NIN2)
      real*8    x1b(*), y1b(*), x2b(*), y2b(*)
      real*8    m1b(*), m2b(*)
      integer NBUOY
 
      real*8 x1u(_NSTMAX_), y1u(_NSTMAX_), m1u(_NSTMAX_)
      real*8 x2u(_NSTMAX_), y2u(_NSTMAX_), m2u(_NSTMAX_)
 
      real*8 x1(_NSTMAX_), y1(_NSTMAX_), m1(_NSTMAX_)
      real*8 x2(_NSTMAX_), y2(_NSTMAX_), m2(_NSTMAX_)
      integer nlist(_NSTMAX_)
      integer NUSEA, NUSEB, NBRS
 
      real*8 dl(3)
      integer nl(3)
      real*8 d1, d2
 
      integer   n, k
      integer   i1, i2
      integer   imin
      integer   m, o
      integer*2 nu(101,101)
      integer*2 n1(101,101,20)
      integer*2 n2(101,101,20)
      integer*2 n3(101,101,20)
 
      integer NIN1u, NIN2u
 
      integer*2 vote(_NVOTE_,_NVOTE_)
      integer   vlist(_NVOTE_)
 
 
      NUSEA = imin(NIN1,_NVOTE_)
      NUSEB = imin(NIN2,_NVOTE_)
 
c
c step 0: copy the brightest _NMATMX_ into (x1,y1,m1) arrays
c
!      print*,'  MM99: copy the _NMATMX_ brightest into new arrays...'
!      print*,'  MM99: _NMATMX: ',_NMATMX_
!      print*,'  MM99:     im1: ',NIN1
      call ord_copy(xin1,yin1,min1,NIN1,x1,y1,m1,_NMATMX_)
!      print*,'  MM99:     im2: ',NIN2
      call ord_copy(xin2,yin2,min2,NIN2,x2,y2,m2,_NMATMX_)
 
 
      do n = 1, NIN1
         x1u(n) = x1(n)
         y1u(n) = y1(n)
         enddo
      NIN1u = NIN1
      do n = 1, NIN2
         x2u(n) = x2(n)
         y2u(n) = y2(n)
         enddo
      NIN2u = NIN2
 
!      print*,'  MM99: NIN1: ',NIN1,NIN1u,m1u(NIN1u)
!      print*,'  MM99: NIN2: ',NIN2,NIN2u,m2u(NIN2u)
 
 
c
c step 1: find the _NMATMX_ brightest in each list
c
!      print*,'  MM99: show the ',_NMATMX_,' brightest...'
 117  format(1x,'  MM99:       N   __M1___ __M2___',1x,
     .                       '__X1___ __Y1___',1x,
     .                       '__X2___ __Y2___')
!      write(*,117)
!      do n = 1,min(max(NIN1,NIN2),9),1
!         write(*,'(''   MM99: '',4x,i5,6f8.2)')
!     .           n,m1(n),m2(n),x1(n),y1(n),x2(n),y2(n)
!         enddo
!      do n = 10,min(max(NIN1,NIN2),90),10
!         write(*,'(''   MM99: '',4x,i5,6f8.2)')
!     .           n,m1(n),m2(n),x1(n),y1(n),x2(n),y2(n)
!         enddo
!      do n = 100,min(max(NIN1,NIN2),500),100
!         write(*,'(''   MM99: '',4x,i5,6f8.2)')
!     .           n,m1(n),m2(n),x1(n),y1(n),x2(n),y2(n)
!         enddo
!      do n = 1000,min(max(NIN1,NIN2),50000),1000
!         write(*,'(''   MM99: '',4x,i5,6f8.2)')
!     .            n,m1(n),m2(n),x1(n),y1(n),x2(n),y2(n)
!         enddo
      n = max(NIN1,NIN2)
 !        write(*,'(''   MM99: '',4x,i5,6f8.2)')
 !    .            n,m1(n),m2(n),x1(n),y1(n),x2(n),y2(n)
 
 
      do i1 = 1, 101
      do i2 = 1, 101
         nu(i1,i2) = 0
         do k = 1, 10
            n1(i1,i2,k) = 0
            n2(i1,i2,k) = 0
            n3(i1,i2,k) = 0
            enddo
         enddo
         enddo
 
c
c step 2: for each of these find the 25 nearest neighbors
c         fill the hash table with vital info
      NBRS = 25
!      print*,'  MM99: fill hash...'
!      print*,'  MM99:     NBRS: ',NBRS
!      print*,'  MM99:    NUSEA: ',NUSEA
!      print*,'  MM99:    NUSEB: ',NUSEB
      do n = 1, NUSEA
!         if (n.eq.n/250*250) print*,'  MM99:    n: ',n
         call fnd_nbrs(x1(n),y1(n),x1,y1,NUSEA,NBRS,nlist)
         do m = 2  , NBRS-1
         do o = m+1, NBRS
            nl(1) = n
            nl(2) = nlist(m)
            nl(3) = nlist(o)
            dl(1) = sqrt((x1(nl(2))-x1(nl(3)))**2+
     .                   (y1(nl(2))-y1(nl(3)))**2)
            dl(2) = sqrt((x1(nl(1))-x1(nl(3)))**2+
     .                   (y1(nl(1))-y1(nl(3)))**2)
            dl(3) = sqrt((x1(nl(2))-x1(nl(1)))**2+
     .                   (y1(nl(2))-y1(nl(1)))**2)
            call cosort2di(dl,nl,3)
            d2 = dl(2) / dl(3)
            i2 = 100*d2*(2*d2-1)+1
            d1 = dl(1) / dl(3)
            i1 = 100*(d1-(1-d2))/(d2-(1-d2))+1
            if (i1.gt.101) goto 5
            if (i2.gt.101) goto 5
            if (i1.lt.001) goto 5
            if (i2.lt.001) goto 5
            do k = 1, nu(i1,i2)
               if (n1(i1,i2,k).eq.nl(1)) goto 5
               enddo
            if (nu(i1,i2).gt.19) goto 5
            nu(i1,i2) = nu(i1,i2)+1
            n1(i1,i2,nu(i1,i2)) = nl(1)
            n2(i1,i2,nu(i1,i2)) = nl(2)
            n3(i1,i2,nu(i1,i2)) = nl(3)
 5          continue
            enddo
            enddo
         enddo
 
 
 
c
c step 3: do same (find 25 nearest...) for the 2nd epoch
c         images, but USE the key to VOTE for coincidences
c
      do i1 = 1, _NVOTE_
      do i2 = 1, _NVOTE_
         vote(i1,i2) = 0
         enddo
         enddo
!      print*,'  MM99: unpack hash...',NUSEA,NUSEB,NBRS
!      print*,'  MM99:     NBRS: ',NBRS
!      print*,'  MM99:    NUSEA: ',NUSEA
!      print*,'  MM99:    NUSEB: ',NUSEB
      do n = 1, NUSEB
!         if (n.eq.n/250*250) print*,'  MM99:    n: ',n
         call fnd_nbrs(x2(n),y2(n),x2,y2,NUSEB,NBRS,nlist)
         do m = 2  , NBRS-1
         do o = m+1, NBRS
            nl(1) = n
            nl(2) = nlist(m)
            nl(3) = nlist(o)
            dl(1) = sqrt((x2(nl(2))-x2(nl(3)))**2+
     .                   (y2(nl(2))-y2(nl(3)))**2)
            dl(2) = sqrt((x2(nl(1))-x2(nl(3)))**2+
     .                   (y2(nl(1))-y2(nl(3)))**2)
            dl(3) = sqrt((x2(nl(2))-x2(nl(1)))**2+
     .                   (y2(nl(2))-y2(nl(1)))**2)
            call cosort2di(dl,nl,3)
            d2 = dl(2) / dl(3)
            i2 = 100*d2*(2*d2-1)+1
            d1 = dl(1) / dl(3)
            i1 = 100*(d1-(1-d2))/(d2-(1-d2))+1
            if (i1.gt.101) goto 6
            if (i2.gt.101) goto 6
            if (i1.lt.001) goto 6
            if (i2.lt.001) goto 6
            do k = 1, nu(i1,i2)
               vote(nl(1),n1(i1,i2,k)) = vote(nl(1),n1(i1,i2,k))+1
               vote(nl(2),n2(i1,i2,k)) = vote(nl(2),n2(i1,i2,k))+1
               vote(nl(3),n3(i1,i2,k)) = vote(nl(3),n3(i1,i2,k))+1
               enddo
 6          continue
            enddo
            enddo
         enddo
 
      NBUOY = 0
!      print*,'  MM99: construct BUOY array...'
      do i1 = 1, NUSEA
         do i2 = 1, NUSEB
            vlist(i2) = vote(i2,i1)
            nlist(i2) = i2
            enddo
         call rcosort2ii(vlist,nlist,NUSEA,10)
         if (vlist(1).gt.2*vlist(2)) then
            NBUOY = NBUOY + 1
            x1b(NBUOY) = x1(i1)
            y1b(NBUOY) = y1(i1)
            m1b(NBUOY) = m1(i1)
            x2b(NBUOY) = x2(nlist(1))
            y2b(NBUOY) = y2(nlist(1))
            m2b(NBUOY) = m2(nlist(1))
            endif
         enddo
      close(146)
 
!      print*,'  MM99:  NBUOY: ',NBUOY
      return
      end
 
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/ROUTINES/MIRMAT/MM5CCHEK.G"
c****                                         
c*********************************************
                                              
 
c-------------------------------------------------
c
c this routine will assume that the scale between the
c two star lists is the same and will search for a match
c subject to that constraint; this removes one degree
c of freedom from the general similar triangles approaches.
c
      subroutine miracle_match5C(xin1,yin1,min1,NIN1,
     .                           xin2,yin2,min2,NIN2,
     .                           x1b,y1b,m1b,
     .                           x2b,y2b,m2b,NBUOY)
      implicit none
 
      integer NIN1
      real*8    xin1(NIN1), yin1(NIN1), min1(NIN1)
      integer NIN2
      real*8    xin2(NIN2), yin2(NIN2), min2(NIN2)
      real*8    x1b(*), y1b(*), x2b(*), y2b(*)
      real*8    m1b(*), m2b(*)
      integer NBUOY
 
      real*8 x1(_NSTMAX_), y1(_NSTMAX_), m1(_NSTMAX_)
      real*8 x2(_NSTMAX_), y2(_NSTMAX_), m2(_NSTMAX_)
 
      integer   n, m
      real*8 dx, dy, dd
 
      real*8    ang, agmx, angr
      real*8    ra, rb
      integer ia, ib
      integer iang
      integer ma, mb, NL, NLs
      integer nfmx, namx
 
      integer nvote(500)
      integer mvote(500)
      integer ivote(500)
      integer lvote(361)
 
      integer msave(250)
      integer lsave(250)
 
      real*8 xl(1000)
      real*8 yl(1000)
      integer na, nb
      real*8    xa, xb
      real*8    ya, yb, PI_314
 
      integer   mask(801,801)
      integer nmask, bmask
 
      integer NUSEA
      integer NUSEB
 
      integer NIN1u, NIN2u
      integer iangu
      
      PI_314 = 4.d0*atan(1.0d0)
 
      NUSEA = 100
      NUSEB = 200
 
!      print*,'MM5CHK: '
!      print*,'MM5CHK:  use brightest stars'
!      print*,'MM5CHK:  '
!      print*,'MM5CHK:   NIN1: ',NIN1,NUSEA
!      print*,'MM5CHK:   NIN2: ',NIN2,NUSEB
!      print*,'MM5CHK:  '
 
      if (NIN1.lt.NUSEA.or.NIN2.lt.NUSEB) then
!         print*,'MM5CHK: '
!         print*,'MM5CHK: You need at least  to '
!         print*,'MM5CHK: find the matches...'
!         print*,'MM5CHK: NIN1: ',NIN1,'  (',NUSEA,') '
!         print*,'MM5CHK: NIN2: ',NIN2,'  (',NUSEB,') '
!         print*,'MM5CHK: '
!         print*,'MM5CHK: (return) '
         return
         endif
 
      NIN1u = NUSEA
      NIN2u = NUSEB
      call ord_brite(xin1,yin1,min1,NIN1,x1,y1,m1,NIN1u,NUSEA)
!      print*,'MM5CHK: ORDBRITE1: ',NIN1,NIN1u
      call ord_brite(xin2,yin2,min2,NIN2,x2,y2,m2,NIN2u,NUSEB)
!      print*,'MM5CHK: ORDBRITE2: ',NIN2,NIN2u
 
 
      do ia = 1, 801
      do ib = 1, 801
         mask(ia,ib) = 0
         enddo
         enddo
      do na = 1, 250
         nmask = 0
         bmask = 0
         do NB = 1, NIN1
            xa = x1(na)
            ya = y1(na)
            xb = xin1(nb)
            yb = yin1(nb)
            dx = xa-xb
            dy = ya-yb
            dd = sqrt(dx**2+dy**2)
            if (dd.lt.195.and.dd.gt.20) then
               ia = 401.0 + dx
               ib = 401.0 + dy
               if (mask(ia,ib).ne.0) then
                  bmask = bmask + 1
                  endif
               if (mask(ia,ib).eq.0) then
                  mask(ia,ib) = na
                  nmask = nmask + 1
                  endif
               endif
            enddo!nb
!         write(*,193) na,nmask,bmask,x1(na),y1(na),m1(na)
 193     format(i3,1x,i3,1x,i3,1x,3f8.2)
         enddo!na
 
      do iang = 1, 361
         lvote(iang) = 0
         enddo
 
!      print*,'MM5CHK:  voting results'
      do ma = 1, 250
         nfmx = 0
         namx = 0
         agmx = 0
         NLs = 0
         do mb = 1, NIN2
            xa = x2(ma)
            ya = y2(ma)
            xb = xin2(mb)
            yb = yin2(mb)
            dx = xa-xb
            dy = ya-yb
            dd = sqrt(dx**2+dy**2)
            if (dd.lt.195.and.dd.gt.20.and.NLs.lt.1000) then
               NLs = NLs + 1
               xl(NLs) = dx
               yl(NLs) = dy
               endif
            enddo!mb
         do n = 1, 500
            ivote(n) = n
            mvote(n) = 0
            enddo
         do iangu = 0000, 3600, 1
            ang = 0.1*iangu
            do n = 1, 250
               nvote(n) = 0
               enddo
            angr = ang/180.d0*PI_314
            do NL = 1, NLs
               dx = xl(NL)*cos(angr)+yl(NL)*sin(angr)
               dy =-xl(NL)*sin(angr)+yl(NL)*cos(angr)
               if (abs(dx).lt.395.and.abs(dy).lt.395) then
               ra = 401+dx
               rb = 401+dy
               do ia = int(ra-1.0), int(ra+1.0)
               do ib = int(rb-1.0), int(rb+1.0)
                  if (mask(ia,ib).ne.0) then
                     na = mask(ia,ib)
                     nvote(na) = nvote(na) + 1
                     if (nvote(na).gt.nfmx) then
                        nfmx = nvote(na)
                        namx = na
                        agmx = ang
                        endif
                     if (nvote(na).gt.mvote(na)) then
                        mvote(na) = nvote(na)
                        endif
                     endif
                  enddo
                  enddo
               endif
               enddo!NL
            enddo!ang
 
         iang = agmx
         if (iang.lt.001) iang = iang + 360
         if (iang.gt.361) iang = iang - 360
 
         call rcosort2ii(mvote,ivote,250,10)
 
         if (agmx.lt.1) agmx = agmx+360.0
         iang = agmx-0.5
         if (iang.lt.001) iang = iang + 360
         if (iang.gt.361) iang = iang - 360
         if (mvote(1).gt.2*mvote(2))
     .   lvote(iang) = lvote(iang)+1
 
 
         iang = agmx+0.5
         if (iang.lt.001) iang = iang + 360
         if (iang.gt.361) iang = iang - 360
         if (mvote(1).gt.2*mvote(2))
     .   lvote(iang) = lvote(iang)+1
 
         iang = agmx
         if (iang.lt.001) iang = iang + 360
         if (iang.gt.361) iang = iang - 360
 
 
c         write(*,194) ma,x2(ma),y2(ma),m2(ma),
c     .                NLs,lvote(iang),iang,
c     .                nfmx,namx,agmx,
c     .                (mvote(n),n=1,10)
c 194     format(10x,i3,1x,3f8.2,1x,
c     .              5i6,1x,f6.1,1x,10i4)
 
         lsave(ma) = iang
         msave(ma) = ivote(1)
 
         if (lvote(iang).ge.5) then
!            print*,'MM5CHK: ---> WE HAVE 10+ VOTES FOR MATCHES!'
!            print*,'MM5CHK: ---> iang: ',iang
!            print*,'MM5CHK: ---> lvot: ',lvote(iang)
            NBUOY = 0
            do m = 1, ma
               if (abs(lsave(m)-iang    ).le.1.or.
     .             abs(lsave(m)-iang-360).le.1.or.
     .             abs(lsave(m)-iang+360).le.1) then
                      NBUOY = NBUOY + 1
                      na = msave(m)
                      x1b(NBUOY) = x1(na)
                      y1b(NBUOY) = y1(na)
                      m1b(NBUOY) = m1(na)
                      x2b(NBUOY) = x2(m )
                      y2b(NBUOY) = y2(m )
                      m2b(NBUOY) = m2(m )
c                      write(*,114) NBUOY,na,m,
c     .                   x1b(NBUOY),y1b(NBUOY),
c     .                   x2b(NBUOY),y2b(NBUOY),
c     .                   m1b(NBUOY),m2b(NBUOY)
c 114                  format(3i5,1x,6f8.2)
                      endif
                enddo
            return
            endif
 
         enddo!ma
 
      NBUOY = 0
 
      return
 
      end
 
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/ROUTINES/MIRMAT/MM_CCHEK_SNS.F"
c****                                         
c*********************************************
                                              
 
c-----------------------------------------------------------
c
c this routine will assume that the scale between the
c two star lists is the same and will search for a match
c subject to that constraint; this removes one degree
c of freedom from the general similar triangles approaches.
c
c
      subroutine miracle_match_SNS(xin1,yin1,min1,NIN1,
     .                             xin2,yin2,min2,NIN2,
     .                             x1b,y1b,m1b,
     .                             x2b,y2b,m2b,NBUOY)
      implicit none
 
      integer NIN1
      real*8    xin1(NIN1), yin1(NIN1), min1(NIN1)
      integer NIN2
      real*8    xin2(NIN2), yin2(NIN2), min2(NIN2)
      real*8    x1b(*), y1b(*), x2b(*), y2b(*)
      real*8    m1b(*), m2b(*)
      integer NBUOY
 
      real*8 x1(_NSTMAX_), y1(_NSTMAX_), m1(_NSTMAX_)
      real*8 x2(_NSTMAX_), y2(_NSTMAX_), m2(_NSTMAX_)
 
      integer   n, m
      real*8 dx, dy, dd
 
      real*8    ang, agmx, angr
      real*8    ra, rb
      integer ia, ib
      integer iang
      integer ma, mb, NL, NLs
      integer nfmx, namx
 
      integer nvote(500)
      integer mvote(500)
      integer ivote(500)
      integer lvote(361)
 
      integer msave(250)
      integer lsave(250)
 
      real*8 xl(1000)
      real*8 yl(1000)
      integer na, nb
      real*8    xa, xb
      real*8    ya, yb
 
      integer   mask(801,801)
      integer nmask, bmask
 
      integer NUSEA
      integer NUSEB
 
      integer NIN1u, NIN2u
      integer iangu
 
      integer n1a, n1b, n1c, n1
      real*8  dx1ba, dy1ba, dd1ba, dm1ba
      real*8  dx1cb, dy1cb, dd1cb
      real*8  dx1ca, dy1ca, dd1ca, dm1ca
      real*8  prd1d
      real*8  prd1x
 
      integer n2a, n2b, n2c, n2
      real*8  dx2ba, dy2ba, dd2ba, dm2ba
      real*8  dx2cb, dy2cb, dd2cb
      real*8  dx2ca, dy2ca, dd2ca, dm2ca
      real*8  prd2d
      real*8  prd2x
 
      real*8  x1u(3), y1u(3)
      real*8  x2u(3), y2u(3)
      real*8  A6, B6, C6, D6, x1o, y1o, x2o, y2o
      real*8  DLTA
      real*8  A3, B3, C3, D3
      real*8  x2e, y2e
      real*8  etot
      integer L
      integer ntot
      integer nmin
 
      real*8  emin
      integer e1a, e1b, e1c
      integer e2a, e2b, e2c
 
      character*30 STRING1
      character*30 STRING2
 
c
c----------------------------------------------------------
c
 
      NUSEA = 25
      NUSEB = 25
 
!      print*,'MM_SNS: '
!      print*,'MM_SNS: ENTER...'
!      print*,'MM_SNS:     NIN1: ',NIN1
!      print*,'MM_SNS:     NIN2: ',NIN2
!      print*,'MM_SNS: '
 
      NIN1u = min(NUSEA,NIN1)
      NIN2u = min(NUSEB,NIN2)
 
      call ord_brite(xin1,yin1,min1,NIN1,x1,y1,m1,NIN1u,NUSEA)
!      print*,'MM_SNS: ORDBRITE1: ',NIN1,NIN1u
      call ord_brite(xin2,yin2,min2,NIN2,x2,y2,m2,NIN2u,NUSEB)
!      print*,'MM_SNS: ORDBRITE2: ',NIN2,NIN2u
 
      do n = 1, max(NIN1u,NIN2u)
         STRING1 = ' '
         STRING2 = ' '
         if (n.le.NIN1u) write(STRING1,129) x1(n),y1(n),m1(n)
         if (n.le.NIN2u) write(STRING2,129) x2(n),y2(n),m2(n)
!         write(*,119) n,STRING1,STRING2
  119    format(1x,'MM_SNS: ',i4.4,5x,a30,5x,a30)
  129    format(3f10.2)
         enddo
 
      emin = 10
      e1a = 0
      e1b = 0
      e1c = 0
      e2a = 0
      e2b = 0
      e2c = 0
 
      ntot = 0
 
!      print*,'MM_SNS: '
!      print*,'MM_SNS: FIND THE BEST THREE STARS WITH 3-PARAM TRANS'
!      print*,'MM_SNS: '
      do n1a = 1,     NIN1u-2
      do n1b = n1a+1, NIN1u-1
      do n1c = n1b+1, NIN1u
         if (n1b.ne.n1a.and.n1c.ne.n1b.and.n1c.ne.n1a) then
             x1u(1) = x1(n1a)
             y1u(1) = y1(n1a)
             x1u(2) = x1(n1b)
             y1u(2) = y1(n1b)
             x1u(3) = x1(n1c)
             y1u(3) = y1(n1c)
             do n2a = 1, NIN2u
             do n2b = 1, NIN2u
             do n2c = 1, NIN2u
                if (n2b.ne.n2a.and.n2c.ne.n2b.and.n2c.ne.n2a) then
                    x2u(1) = x2(n2a)
                    y2u(1) = y2(n2a)
                    x2u(2) = x2(n2b)
                    y2u(2) = y2(n2b)
                    x2u(3) = x2(n2c)
                    y2u(3) = y2(n2c)
                    call glob_fit6nrDP(x1u,y1u,x2u,y2u,3,
     .                                 A6,B6,C6,D6,x1o,y1o,x2o,y2o)
                    DLTA = A6*D6-B6*C6
                    if (DLTA.gt.0) then
                       A3 = (A6+D6)/2/sqrt(DLTA)
                       B3 = (B6-C6)/2/sqrt(DLTA)
                       C3 = (C6-B6)/2/sqrt(DLTA)
                       D3 = (A6+D6)/2/sqrt(DLTA)
                      !A3 = (A6+D6)/2
                      !B3 = (B6-C6)/2
                      !C3 = (C6-B6)/2
                      !D3 = (A6+D6)/2
                       etot = 0.
                       do L = 1, 3
                          x2e = x2o + A3*(x1u(L)-x1o) + B3*(y1u(L)-y1o)
                          y2e = y2o + C3*(x1u(L)-x1o) + D3*(y1u(L)-y1o)
                          etot = etot + (x2u(L)-x2e)**2
     .                                + (y2u(L)-y2e)**2
                          enddo
                       etot = sqrt(etot/2)
                       if (etot.lt.5.0) then
                           ntot = 0
                           do n1 = 1, NIN1u
                              x2e = x2o + A3*(x1(n1)-x1o)
     .                                  + B3*(y1(n1)-y1o)
                              y2e = y2o + C3*(x1(n1)-x1o)
     .                                  + D3*(y1(n1)-y1o)
                              do n2 = 1, NIN2u
                                 if (sqrt((x2(n2)-x2e)**2
     .                                   +(y2(n2)-y2e)**2).lt.5)
     .                                        ntot = ntot + 1
                                 enddo
                              enddo
                           if (ntot.gt.nmin.or.
     .                         (etot.lt.emin.and.ntot.ge.nmin)) then
                               emin = etot
                               nmin = ntot
                               e1a = n1a
                               e1b = n1b
                               e1c = n1c
                               e2a = n2a
                               e2b = n2b
                               e2c = n2c
!                               write( *,191) n1a, n1b, n1c,
!     .                                       n2a, n2b, n2c,
!     .                                       etot, ntot, DLTA,
!     .                                       A6, B6, C6, D6,
!     .                                       x1o, y1o, x2o, y2o,
!     .                                       A3, B3, C3, D3
                               endif
                           endif
                       endif
                    endif
                enddo
                enddo
                enddo
             endif
          enddo
          enddo
          enddo
 
  999 continue
!      print*,'MM_SNS: '
!      write( *,191) e1a, e1b, e1c, e2a, e2b, e2c, emin, nmin
!      print*,'MM_SNS: '
 
      NBUOY = 0
      if (emin.lt.5.0) then
          NBUOY = 3
          x1b(1) = x1(e1a)
          y1b(1) = y1(e1a)
          m1b(1) = m1(e1a)
          x1b(2) = x1(e1b)
          y1b(2) = y1(e1b)
          m1b(2) = m1(e1b)
          x1b(3) = x1(e1c)
          y1b(3) = y1(e1c)
          m1b(3) = m1(e1c)
          x2b(1) = x2(e2a)
          y2b(1) = y2(e2a)
          m2b(1) = m2(e2a)
          x2b(2) = x2(e2b)
          y2b(2) = y2(e2b)
          m2b(2) = m2(e2b)
          x2b(3) = x2(e2c)
          y2b(3) = y2(e2c)
          m2b(3) = m2(e2c)
          endif
 
 
      return
 
  191 format(1x,'MM_SNS: ',1x,i3.3,1x,i3.3,1x,i3.3,
     .                     3x,i3.3,1x,i3.3,1x,i3.3,
     .                     3x,f5.2,3x,i3,3x,f7.4,
     .                     5x,4f8.4,5x,4f9.2,
     .                     5x,4f8.4)
 
      end
 
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/ROUTINES/MIRMAT/INCORPBM.F"
c****                                         
c*********************************************
                                              
 
c---------------------------------------------------------
c
c once we have an initial match, this will use that match
c to incoroporate more stars (bright and faint) into a
c more comprehensive list of matches
c
      subroutine incorp_buoy(x1bt,y1bt,x2bt,y2bt,NBT,
     .                       x1,y1,m1,NIN1,
     .                       x2,y2,m2,NIN2,
     .                       x1b,y1b,m1b,
     .                       x2b,y2b,m2b,NBUOY,DISC)
      implicit none
 
      real*8 x1bt(*), y1bt(*)
      real*8 x2bt(*), y2bt(*)
      integer NBT
      real*8 x1(*), y1(*), m1(*)
      integer NIN1
      real*8 x2(*), y2(*), m2(*)
      integer NIN2
      real*8 x1b(*), y1b(*), m1b(*)
      real*8 x2b(*), y2b(*), m2b(*)
      integer NBUOY
      real*8 DISC
 
      integer ix, iy, ixc, iyc
c
c hash table to match things up more easily
c
      integer hash(2001,2001)
      integer NCONFL
 
      real*8 mo
      integer n1, n2
      real*8 x2g, y2g
      real*8 dmin, duse
      real*8 dminx, dminy
      integer nmin
 
      real*8 minarr, maxarr
      real*8 xlo, xhi
      real*8 ylo, yhi
      real*8 AG,BG,CG,DG,Xo,Yo,Wo,Zo
 
      NCONFL = 0
 
!      print*,'INCBUY: '
!      print*,'INCBUY:   ENTER INCORPBM.F: '
!      print*,'INCBUY:    NBT: ',NBT
!      print*,'INCBUY:   NIN1: ',NIN1
!      print*,'INCBUY:   NIN2: ',NIN2
!      print*,'INCBUY:  NBUOY: ',NBUOY
!      print*,'INCBUY:   DISC: ',DISC
!      print*,'INCBUY: '
      if (NBT.lt.20) then
         do ix = 1,NBT
!            write(*,'('' INCBUY: '',4f10.2)') x1bt(ix), y1bt(ix),
!     .                                   x2bt(ix), y2bt(ix)
            enddo
         endif
!      print*,'INCBUY: '
 
      call glob_fit6nrDP(x1bt,y1bt,x2bt,y2bt,NBT,
     .                   AG, BG, CG, DG, Xo, Yo, Wo, Zo)
!      write( *,1333) AG,BG,CG,DG,Xo-Wo,Yo-Zo,NBT
! 1333 format(' INCBUY: ',3x,4(f10.6,1x),3x,2(f9.3,1x),3x,i6)
!      print*,'INCBUY: Xo = ',Xo,Yo
!      print*,'INCBUY: Wo = ',Wo,Zo
!      print*,'INCBUY:      '
!      write(*,'('' INCBUY: AG = '',f10.6)') AG
!      write(*,'('' INCBUY: BG = '',f10.6)') BG
!      write(*,'('' INCBUY: CG = '',f10.6)') CG
!      write(*,'('' INCBUY: DG = '',f10.6)') DG
!      write(*,'('' INCBUY: dX = '',3f10.2)') Xo,Wo,Xo-Wo
!      write(*,'('' INCBUY: dY = '',3f10.2)') Yo,Zo,Yo-Zo
!      write(*,'('' INCBUY:  '')')
 
      do ix = 1, 2001
      do iy = 1, 2001
         hash(ix,iy) = 0
         enddo
         enddo
 
      xlo = minarr(x2,NIN2)
      xhi = maxarr(x2,NIN2)
      ylo = minarr(y2,NIN2)
      yhi = maxarr(y2,NIN2)
 
!      print*,'INCBUY:  RANGES:'
!      print*,'INCBUY:       x: ',minarr(x1,NIN1),maxarr(x1,NIN1)
!      print*,'INCBUY:       y: ',minarr(y1,NIN1),maxarr(y1,NIN1)
!      print*,'INCBUY:       m: ',minarr(m1,NIN1),maxarr(m1,NIN1)
!      print*,'INCBUY:      xx: ',minarr(x2,NIN2),maxarr(x2,NIN2)
!      print*,'INCBUY:      yy: ',minarr(y2,NIN2),maxarr(y2,NIN2)
!      print*,'INCBUY:      mm: ',minarr(m2,NIN2),maxarr(m2,NIN2)
!      print*,'INCBUY:     xhi: ',xhi,xlo
!      print*,'INCBUY:     yhi: ',yhi,ylo
 
      if (xhi.lt.xlo+2000) xhi = xlo+2000
      if (yhi.lt.ylo+2000) yhi = ylo+2000
 
      do n2 = 1, NIN2
         ix = 1 + 2000*((x2(n2)+0.5)-xlo)/(xhi-xlo)
         iy = 1 + 2000*((y2(n2)+0.5)-ylo)/(yhi-ylo)
         if (ix.ge.1.and.ix.le.2001.and.
     .       iy.ge.1.and.iy.le.2001) then
                mo =  99.0
                mo = 999.0
                if (hash(ix,iy).ne.0) NCONFL = NCONFL + 1
                if (hash(ix,iy).ne.0) mo = m2(hash(ix,iy))
                if (m2(n2).lt.mo) hash(ix,iy) = n2
             endif
         enddo
 
      NBUOY = 0
!      print*,'INCBUY: INCORPBM.F'
!      print*,'INCBUY:    NBT: ',NBT
!      print*,'INCBUY:   NCON: ',NCONFL
!      print*,'INCBUY:   DISC: ',DISC
!      print*,'INCBUY:   ABCD: ',AG,BG,CG,DG
 
      do n1 = 1, NIN1
         x2g = Wo + AG*(x1(n1)-Xo) + BG*(y1(n1)-Yo)
         y2g = Zo + CG*(x1(n1)-Xo) + DG*(y1(n1)-Yo)
         ixc = 1 + 2000*(x2g-xlo)/(xhi-xlo)
         iyc = 1 + 2000*(y2g-ylo)/(yhi-ylo)
         dmin  = DISC
         dminx = DISC
         dminy = DISC
         nmin = 0
         do ix = int(max(ixc-max(DISC,2.),   1.)-0.5),
     .           int(min(ixc+max(DISC,2.),2001.)+0.5)
         do iy = int(max(iyc-max(DISC,2.),   1.)-0.5),
     .           int(min(iyc+max(DISC,2.),2001.)+0.5)
            if (hash(ix,iy).ne.0) then
               n2 = hash(ix,iy)
               duse = abs(x2g-x2(n2)) + abs(y2g-y2(n2))
               if (duse.lt.dmin) then
                  dmin = duse
                  dminx = x2g-x2(n2)
                  dminy = y2g-y2(n2)
                  nmin = n2
                  endif
               endif
            enddo
            enddo
!         if ((n1.eq.n1/1000*1000).or.n1.lt.0010.or.
!     .       (n1.eq.n1/0100*0100.and.n1.lt.1000).or.
!     .       (n1.eq.n1/0010*0010.and.n1.lt.0100).or.
!     .       (n1.lt.25))
!     .       write(*,119) n1,NIN1,NBUOY,NBT,
!     .          x1(n1),y1(n1),x2g,y2g,x2g,y2g,dmin,nmin
 119     format(' INCBUY: ',4i8,1x,6f9.2,1x,f6.3,1x,i4)
         if (nmin.ne.0.and.dmin.lt.DISC) then
            NBUOY = NBUOY + 1
            x1b(NBUOY) = x1(n1)
            y1b(NBUOY) = y1(n1)
            m1b(NBUOY) = m1(n1)
            x2b(NBUOY) = x2(nmin)
            y2b(NBUOY) = y2(nmin)
            m2b(NBUOY) = m2(nmin)
            if (NBUOY.gt._NMATMX_) then
               print*,'INCBUY: NEED TO INCREASE _NMATMX_'
               print*,'INCBUY: ABOVE ',_NMATMX_
               stop
               endif
            endif
         enddo
 
      xlo = minarr(x2b,NBUOY)
      xhi = maxarr(x2b,NBUOY)
      ylo = minarr(y2b,NBUOY)
      yhi = maxarr(y2b,NBUOY)
 
!      print*,'INCBUY: '
!      print*,'INCBUY:   x: ',minarr(x1b,NBUOY),maxarr(x1b,NBUOY)
!      print*,'INCBUY:   y: ',minarr(y1b,NBUOY),maxarr(y1b,NBUOY)
!      print*,'INCBUY:   m: ',minarr(m1b,NBUOY),maxarr(m1b,NBUOY)
!      print*,'INCBUY:  xx: ',minarr(x2b,NBUOY),maxarr(x2b,NBUOY)
!      print*,'INCBUY:  yy: ',minarr(y2b,NBUOY),maxarr(y2b,NBUOY)
!      print*,'INCBUY:  mm: ',minarr(m2b,NBUOY),maxarr(m2b,NBUOY)
!      print*,'INCBUY: '
!      print*,'INCBUY: ---> NBUOY: ',NBUOY
!      print*,'INCBUY: ---> exit IBT...'
!      print*,'INCBUY: '
 
      if (N1.lt.50.and.N2.lt.50) then
!         print*,'INCBUY:    '
!         print*,'INCBUY: --- N1: '
         do N1 = 1, NIN1
!            write(*,'('' INCBUY:  '',i4,1x,2f9.2,1x,f8.3)')
!     .                               N1,x1(n1),y1(n1),m1(n1)
            enddo
!         print*,'INCBUY:    '
!         print*,'INCBUY: --- N2: '
         do N2 = 1, NIN2
!            write(*,'('' INCBUY:  '',i4,1x,2f8.2,1x,f8.3)')
!     .                 N2,x2(n2),y2(n2),m2(n2)
            enddo
         endif
!      print*,'INCBUY: '
!      print*,'INCBUY: '
 
      return
      end
 
 
 
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/ROUTINES/MIRMAT/PURIFYB.F"
c****                                         
c*********************************************
                                              
 
c-------------------------------------------------
c
c this routine will take a list of matched stars and
c will throw out the ones (one at a time) that are not
c consistent to within DMAX
c
      subroutine purify_buoy(x1b,y1b,x2b,y2b,NB,DMAX)
      implicit none
 
      real*8 x1b(*), y1b(*)
      real*8 x2b(*), y2b(*)
      integer NB
      real*8 DMAX
 
 
      real*8 x2g, y2g
      integer n
      real*8 dx, dy, dd
 
      real*8    dw
      integer nw
      real*8    ddl(_NSTMAX_)
      integer NBg
      real*8    DLIM
 
      real*8 AG, BG, CG, DG
      real*8 Xo, Yo, Wo, Zo
      real*8 var
 
!      print*,'PURIFY:  '
!      print*,'PURIFY:  ENTER PURIFY_BUOY --- '
!      print*,'PURIFY:          '
!      print*,'PURIFY:  NBUOY: ',NB
!      print*,'PURIFY:   DMAX: ',DMAX
 
      call glob_fit6nrDP(x1b,y1b,x2b,y2b,NB,
     .                   AG,BG,CG,DG,Xo,Yo,Wo,Zo)
 
 
 1    continue
      call glob_fit6nrDP(x1b,y1b,x2b,y2b,NB,
     .                   AG,BG,CG,DG,Xo,Yo,Wo,Zo)
      dw = 0
      nw = 0
      var = 0
      do n = 1, NB
         x2g = Wo + AG*(x1b(n)-Xo) + BG*(y1b(n)-Yo)
         y2g = Zo + CG*(x1b(n)-Xo) + DG*(y1b(n)-Yo)
         dx = x2b(n)-x2g
         dy = y2b(n)-y2g
         dd = sqrt(dx**2+dy**2)
         ddl(n) = dd
         var = var + dd**2
         if (dd.gt.dw) then
            dw = dd
            nw =  n
            endif
!         if (NB.lt.10) write(*,'('' PURIFY: '',i3,1x,3f8.2)')
!     .                       n,dx,dy,dd
         enddo
      var = sqrt(var/NB)
 
      DLIM = max(0.9*dw,DMAX)
      NBg = 0
      do n = 1, NB
         if (ddl(n).lt.DLIM) then
            NBg = NBg + 1
            x1b(NBg) = x1b(n)
            y1b(NBg) = y1b(n)
            x2b(NBg) = x2b(n)
            y2b(NBg) = y2b(n)
            endif
          enddo
      if (NBg.lt.NB) then ! we rejected one!
!         print*,'PURIFY:  REJECT   DLIM= ',DLIM,'  NBg ---> ',NBg,NB
         NB = NBg
         goto 1
         endif
 
      NB = NBg
 
!      print*,'PURIFY:  '
!      print*,'PURIFY:  EXIT WITH NB = ',NB
!      print*,'PURIFY:  '
 
      return
      end
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/ROUTINES/MIRMAT/PURGBUOY.F"
c****                                         
c*********************************************
                                              
 
 
      subroutine purge_buoyR(x1b,y1b,m1b,
     .                       x2b,y2b,m2b,NB,NIT)
      implicit none
 
      real*8 x1b(*), y1b(*), m1b(*)
      real*8 x2b(*), y2b(*), m2b(*)
      integer NB
      integer NIT
 
      real*8 darray(25)
      data darray /99.0d0,50.00d0,25.00d0,15.0d0, 10.00d0,
     .             8.00d0, 5.00d0, 3.50d0, 2.00d0, 1.50d0,
     .             1.20d0, 1.00d0, 0.80d0, 0.60d0, 0.45d0,
     .             0.35d0, 0.25d0, 0.20d0, 0.16d0, 0.12d0,
     .             0.09d0, 0.07d0, 0.05d0, 0.04d0, 0.03d0/
 
      real*8 DMAX
      real*8 x2o, y2o
      integer n
      real*8 dx, dy, dd, varu
      integer NB0
      real*8 AG,BG,CG,DG,Xo,Yo,Wo,Zo
      integer h
      integer hist(26)
      real*8 x1h(_NMATMX_), y1h(_NMATMX_), m1h(_NMATMX_)
      real*8 x2h(_NMATMX_), y2h(_NMATMX_), m2h(_NMATMX_)
c      character*20 FILE
 
      NB0 = NB
      if (NB.eq.0) stop
 
c      write(FILE,'(''PURGBUOY.IN_.'',i2.2)') NIT
c      open(39,file=FILE,status='unknown')
c      do n = 1, NB
c         write(39,139) x1b(n),y1b(n),
c     .                 x2b(n),y2b(n)
c 139     format(4f9.2)
c         enddo
c      close(39)
 
      call glob_fit6nrDP(x1b,y1b,x2b,y2b,NB,
     .                   AG,BG,CG,DG,Xo,Yo,Wo,Zo)
 
c      print*,'ABCD: ',AG,BG,CG,DG
c      print*,' DXY: ',Xo-Wo,Yo-Zo
 
      do n = 1, NB0
         x1h(n) = x1b(n)
         y1h(n) = y1b(n)
         m1h(n) = m1b(n)
         x2h(n) = x2b(n)
         y2h(n) = y2b(n)
         m2h(n) = m2b(n)
         enddo
 
      do h = 1, 26
         hist(h) = 0
         enddo
      DMAX = darray(NIT)
 
      NB  = 0
      varu = 0.
      do n = 1, NB0
         x2o = Wo + AG*(x1b(n)-Xo) + BG*(y1b(n)-Yo)
         y2o = Zo + CG*(x1b(n)-Xo) + DG*(y1b(n)-Yo)
         dx = x2o-x2b(n)
         dy = y2o-y2b(n)
         dd = sqrt(dx**2+dy**2)
c         write(45,145) x1b(n),y1b(n),x2b(n),y2b(n),dx,dy
c 145     format(1x,10f12.3)
         if (dd.lt.DMAX) then
             NB = NB + 1
             x1b(NB) = x1h(n)
             y1b(NB) = y1h(n)
             m1b(NB) = m1h(n)
             x2b(NB) = x2h(n)
             y2b(NB) = y2h(n)
             m2b(NB) = m2h(n)
             varu = varu + dd**2
             endif
         do h = 1, 25
            if (dd.gt.darray(h)) goto 5
            enddo
         h = 26
 5       hist(h) = hist(h) + 1
         enddo
      varu = sqrt(varu/(NB-1))
 
!      if (NIT.eq.1)
!     .write(*,114) (darray(h),h=25,1,-1)
!      write(*,115) NIT,darray(NIT),NB,
!     .             (hist(h),h=26,1,-1)
 114  format(' PRGBUY: ','NIT',1x,' DMAX ',1x,26f5.2)
 115  format(' PRGBUY: ',1x,i2,1x,f6.2    ,1x,i5,1x,26i5)
 
 
 
      return
      end
 
 
 
 
 
 
      subroutine purge_buoy(x1b,y1b,m1b,
     .                      x2b,y2b,m2b,NB,NIT)
      implicit none
 
      real*8 x1b(*), y1b(*), m1b(*)
      real*8 x2b(*), y2b(*), m2b(*)
      integer NB
      integer NIT
 
      real*8 darray(25)
      data darray /99.0d0,50.00d0,25.00d0,15.0d0, 10.00d0,
     .             8.00d0, 5.00d0, 3.50d0, 2.00d0, 1.50d0,
     .             1.20d0, 1.00d0, 0.80d0, 0.60d0, 0.45d0,
     .             0.35d0, 0.25d0, 0.20d0, 0.16d0, 0.12d0,
     .             0.09d0, 0.07d0, 0.05d0, 0.04d0, 0.03d0/
 
      real*8 DMAX
      real*8 x1o, y1o
      integer n
      real*8 dx, dy, dd, varu
      integer NB0
      real*8 AG,BG,CG,DG,Xo,Yo,Wo,Zo
      integer h
      integer hist(26)
      real*8 x1h(_NMATMX_), y1h(_NMATMX_), m1h(_NMATMX_)
      real*8 x2h(_NMATMX_), y2h(_NMATMX_), m2h(_NMATMX_)
c     character*20 FILE
 
 
!      character*200 PB_STRING(50)
!      integer       PB_STRINGs
 
!      common/PB_STRING/PB_STRINGs,PB_STRING
 
c---------------------------------------------
 
 
      NB0 = NB
      if (NB.eq.0) return
      if (NB.eq.0) stop
 
 
      call glob_fit6nrDP(x2b,y2b,x1b,y1b,NB,
     .                   AG,BG,CG,DG,Xo,Yo,Wo,Zo)
 
 
      do n = 1, NB0
         x1h(n) = x1b(n)
         y1h(n) = y1b(n)
         m1h(n) = m1b(n)
         x2h(n) = x2b(n)
         y2h(n) = y2b(n)
         m2h(n) = m2b(n)
         enddo
 
      do h = 1, 26
         hist(h) = 0
         enddo
      DMAX = darray(NIT)
 
      NB  = 0
      varu = 0.
      do n = 1, NB0
         x1o = Wo + AG*(x2b(n)-Xo) + BG*(y2b(n)-Yo)
         y1o = Zo + CG*(x2b(n)-Xo) + DG*(y2b(n)-Yo)
         dx = x1o-x1b(n)
         dy = y1o-y1b(n)
         dd = sqrt(dx**2+dy**2)
         if (dd.lt.DMAX) then
             NB = NB + 1
             x1b(NB) = x1h(n)
             y1b(NB) = y1h(n)
             m1b(NB) = m1h(n)
             x2b(NB) = x2h(n)
             y2b(NB) = y2h(n)
             m2b(NB) = m2h(n)
             varu = varu + dd**2
             endif
         do h = 1, 25
            if (dd.gt.darray(h)) goto 5
            enddo
         h = 26
 5       hist(h) = hist(h) + 1
         enddo
      varu = sqrt(varu/(NB-1))
 
!      if (NIT.eq.1) then
!         PB_STRINGs = 0
!         write(*,113) (h,h=25,1,-1)
!         write(*,114) (darray(h),h=25,1,-1)
!         PB_STRINGs = PB_STRINGs + 1
!         write(PB_STRING(PB_STRINGs),113) (h,h=25,1,-1)
!         PB_STRINGs = PB_STRINGs + 1
!         write(PB_STRING(PB_STRINGs),114) (darray(h),h=25,1,-1)
!         endif
!      write(*,115) NIT,darray(NIT),NB,
!     .             (hist(h),h=26,int(max(1,NIT-1*0)),-1)
!      PB_STRINGs = PB_STRINGs+1
!      write(PB_STRING(PB_STRINGs),115) NIT,darray(NIT),NB,
!     .             (hist(h),h=26,int(max(1,NIT-1*0)),-1)
 113  format(' PRGBUY:    ',1x,'     ',1x,'     ',1x,26(2x,i2,1x))
 114  format(' PRGBUY: NIT',1x,' DMAX',1x,' NB  ',1x,26f5.2)
 115  format(' PRGBUY: ',   1x,i2,1x,f5.2,1x,i5,1x,26i5)
 
      return
      end
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/ROUTINES/MIRMAT/ORDBRITE.F"
c****                                         
c*********************************************
                                              
 
      subroutine ord_brite(xi,yi,mi,NINi,
     .                     xo,yo,mo,NINu,NINo)
      implicit none
 
      real*8 xi(*), yi(*), mi(*)
      integer NINi
      real*8 xo(*), yo(*), mo(*)
      integer NINu
      integer NINo
 
      integer n
      integer mhist(101), mh, mc
      real*8 mmax, mmin, muse
 
c     print*,'ORDBRT: NINi: ',NINi
c     print*,'ORDBRT: NINo: ',NINo
 
      muse = 99.9
      if (NINo.gt.NINi) NINo = NINi
 
      mmax = -99.0
      mmin =  99.0
      do n = 1, NINi
         if (mi(n).gt.mmax) mmax = mi(n)
         if (mi(n).lt.mmin) mmin = mi(n)
         enddo
c     print*,'ORDBRT: mmax: ',mmax
c     print*,'ORDBRT: mmin: ',mmin
 
      do mh = 1, 101
         mhist(mh) = 0
         enddo
      do n = 1, NINi
         mh = 1 + (mi(n)-mmin)/(mmax-mmin)*100
         mhist(mh) = mhist(mh) + 1
         enddo
 
      mh = 0
      mc = 0
 1    continue
      mh = mh + 1
      mc = mc + mhist(mh)
      if (mc.lt.NINo) goto 1
      muse = mmin + (mh+1.5)/100.*(mmax-mmin)
 
      NINu = 0
      do n = 1, NINi
         if (mi(n).lt.muse) then
            NINu = NINu + 1
            xo(NINu) = xi(n)
            yo(NINu) = yi(n)
            mo(NINu) = mi(n)
            endif
         enddo
 
c     print*,'ORDBRT: NINu: ',NINu,muse,mc
      call cosort3d(mo,xo,yo,NINu)
 
      return
      end
 
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/ROUTINES/TRANS/LOCAL/fnd_nbrs.f"
c****                                         
c*********************************************
                                              
 
c---------------------------------------------------
c
c will find the nearest NLIM stars to a reference position
c
      subroutine fnd_nbrs(xref,yref,x,y,NTOT,NLIM,nl)
      implicit none
 
      real*8 xref, yref
      real*8 x(*), y(*)
      integer NTOT
      integer NLIM
      integer nl(_NSTMAX_)
 
      real    dl(_NSTMAX_)
      integer n, nit
      logical change
 
      do n = 1, NTOT
         dl(n) = (x(n)-xref)**2+(y(n)-yref)**2
         nl(n) = n
         enddo
 
      nit = 0
  1   continue
      change = .false.
      nit = nit + 1
      do n = NTOT-1, 1, -1
         if (dl(n).gt.dl(n+1)) then
            call swap(dl(n),dl(n+1))
            call swip(nl(n),nl(n+1))
            change = .true.
            endif
         enddo
      if (change.and.nit.lt.NLIM) goto 1
 
      return
      end
 
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/ROUTINES/STRING/addstem.f"
c****                                         
c*********************************************
                                              
 
      subroutine addstem(STRING,SUFFIX)
      implicit none
      character*80 STRING
      character*3  SUFFIX
      integer i
 
      do i = 1, 27
         if (STRING(i:i).eq.'.') then
            string = string(1:i) // suffix
            return
            endif
         enddo
 
      print*,'ADDSTEM FAILURE: '
      print*,' '
      print*,'   NO DOT IN STRING: '
      print*,'   STRING: ',STRING
      print*,'   SUFFIX: ',SUFFIX
      print*,' '
      print*,' '
      stop
      end
 
 
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/ROUTINES/STAT/modelist.f"
c****                                         
c*********************************************
                                              
c
c find the window that contains PCT
c find the modpoint of this window
c average all points within 1.5 windows of this
c point
c
      real function modelist(list,N,pct)
      implicit none
 
      integer N
      real list(N)
      real pct
 
      real*8 hold(20000)
      integer i, ip
      integer iwin
      real    dwin
      real    mwin, mmin, mmax
      real    htot
      integer nuse
 
 
      if (N.gt.20000) then
         print*,'modelist: ',N
         stop
         endif
 
      do i = 1, N
         hold(i) = list(i)
         enddo
 
      call dbubble(hold,N)
 
      !print*,'hold(0.5): ',hold(N/2)
 
      ip = pct*N + 0.75
      !print*,'        N: ',N
      !print*,'       ip: ',ip
 
      iwin = 1
      dwin = hold(1+ip)-hold(1)
      do i = 2, N-ip
         if (hold(i+ip)-hold(i).lt.dwin) then
            iwin = i
            dwin = hold(i+ip)-hold(i)
            endif
         enddo
      mwin = 0.5*(hold(iwin)+hold(iwin+ip))
      mmin = mwin - 0.75*dwin
      mmax = mwin + 0.75*dwin
 
      nuse = 0.
      htot = 0
      do i = 1, N
         if (hold(i).ge.mmin.and.hold(i).le.mmax) then
            nuse = nuse + 1
            htot = htot + hold(i)
            endif
         enddo
      modelist = htot / nuse
      !print*,'htot: ',htot
      !print*,'nuse: ',nuse
      !print*,'mode: ',modelist
 
      return
      end
 


      



      


                                                    
 
c--------------------------------------
c
c returns the arctangent in degrees
c
      real*8 function atan2dj(x,y)
      implicit none
      real*8 x, y
      real*8 arad
 
      real*8 xr, yr, PI_314

      PI_314 = 4.d0*atan(1.0d0)
      
      xr = x
      yr = y
      arad = atan2(xr,yr)
      atan2dj = 180.0d0/PI_314*arad
 
      return
      end

                                              
 
      subroutine cosort2ri(vl,nl,NTOT)
      implicit none
 
      real    vl(*)
      integer nl(*)
      integer NTOT
 
      integer n, nit
      logical change
 
      nit = 0
  1   continue
      change = .false.
      nit = nit + 1
      do n = NTOT-1, 1, -1
         if (vl(n).gt.vl(n+1)) then
            call swap(vl(n),vl(n+1))
            call swip(nl(n),nl(n+1))
            change = .true.
            endif
         enddo
      if (change.and.nit.lt.NTOT) goto 1
 
      return
      end
 
      subroutine cosort2di(vl,nl,NTOT)
      implicit none
 
      real*8 vl(*)
      integer nl(*)
      integer NTOT
 
      integer n, nit
      logical change
 
      nit = 0
  1   continue
      change = .false.
      nit = nit + 1
      do n = NTOT-1, 1, -1
         if (vl(n).gt.vl(n+1)) then
            call swdp(vl(n),vl(n+1))
            call swip(nl(n),nl(n+1))
            change = .true.
            endif
         enddo
      if (change.and.nit.lt.NTOT) goto 1
 
      return
      end
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/ROUTINES/SORT/cosrt2ii.f"
c****                                         
c*********************************************
                                              
 
 
      subroutine rcosort2ii(vl,nl,NTOT,NITMAX)
      implicit none
 
      integer vl(*)
      integer nl(*)
      integer NTOT
      integer NITMAX
 
      integer n, nit
      logical change
 
      nit = 0
  1   continue
      change = .false.
      nit = nit + 1
      do n = NTOT-1,1,-1
         if (vl(n).lt.vl(n+1)) then
            call swip(vl(n),vl(n+1))
            call swip(nl(n),nl(n+1))
            change = .true.
            endif
         enddo
      if (change.and.nit.lt.NITMAX) goto 1
 
      return
      end
 
      subroutine rcosort2iii(vl,nl,ml,NTOT,NITMAX)
      implicit none
 
      integer vl(*)
      integer nl(*)
      integer ml(*)
      integer NTOT
      integer NITMAX
 
      integer n, nit
      logical change
 
      nit = 0
  1   continue
      change = .false.
      nit = nit + 1
      do n = NTOT-1,1,-1
         if (vl(n).lt.vl(n+1)) then
            call swip(vl(n),vl(n+1))
            call swip(nl(n),nl(n+1))
            call swip(ml(n),ml(n+1))
            change = .true.
            endif
         enddo
      if (change.and.nit.lt.NITMAX) goto 1
 
      return
      end
 
 
      subroutine rcosort2iiii(vl,nl,ml,ol,NTOT,NITMAX)
      implicit none
 
      integer vl(*)
      integer nl(*)
      integer ml(*)
      integer ol(*)
      integer NTOT
      integer NITMAX
 
      integer n, nit
      logical change
 
      nit = 0
  1   continue
      change = .false.
      nit = nit + 1
      do n = NTOT-1,1,-1
         if (vl(n).lt.vl(n+1)) then
            call swip(vl(n),vl(n+1))
            call swip(nl(n),nl(n+1))
            call swip(ml(n),ml(n+1))
            call swip(ol(n),ol(n+1))
            change = .true.
            endif
         enddo
      if (change.and.nit.lt.NITMAX) goto 1
 
      return
      end
 
c
c
c sorts in ASCENDING ORDER!!!
c
      subroutine cosort3d(r1,r2,r3,NTOT)
      implicit none
      real*8 r1(*), r2(*), r3(*)
      integer NTOT
      integer n
      logical change
 777  continue
      change = .false.
      do n = 1, NTOT-1
         if (r1(n).gt.r1(n+1)) then
            call swdp(r1(n),r1(n+1))
            call swdp(r2(n),r2(n+1))
            call swdp(r3(n),r3(n+1))
            change = .true.
            endif
         enddo
      if (change) goto 777
      end
  
c--------------------------------------
c
c bubble-sorts a real*8 list into ascending order
c
      subroutine dbubble(r1,NTOT)
      implicit none
      real*8 r1(*)
      integer NTOT
      integer n
      real*8 temp
      logical change
 777  continue
      change = .false.
      do n = 1, NTOT-1
         if (r1(n).gt.r1(n+1)) then
            temp = r1(n)
            r1(n) = r1(n+1)
            r1(n+1) = temp
            change = .true.
            endif
         enddo
      if (change) goto 777
      end
 
                                               
 
c--------------------------------------------------------------
c
c this routine does a least squares solution (with no
c data rejection or weighting) for the 6-param linear
c fit for:
c
c
c    x2 = A*(x1-x1o) + B*(y1-y1o) + x2o
c    y2 = C*(x1-x1o) + D*(y1-y1o) + y2o
c
c it may look like there are 8 params, but two of the
c offsets are arbitrary and are just set to the centroid
c of the distribution.
c
c
 
      subroutine glob_fit6nrDP(x1,y1,x2,y2,NUSE,
     .                         A,B,C,D,x1o,y1o,x2o,y2o)
      implicit none
 
      real*8 x1(*), y1(*)
      real*8 x2(*), y2(*)
      integer NUSE
      real*8 A, B, C, D
      real*8 x1o, y1o, x2o, y2o
 
      real*8 sxx, sx, swx, szx
      real*8 syy, sy, swy, szy
      real*8 sw,  sz, sxy
      real*8 dlta
      real*8 dsxx, dsyy, dsxy
      real*8 dswx, dswy, dszx, dszy
 
      integer n
 
      if (NUSE.lt.3) goto 999
 
      x1o = 0
      y1o = 0
      x2o = 0
      y2o = 0
      do n = 1, NUSE
         x1o = x1o + x1(n)
         y1o = y1o + y1(n)
         x2o = x2o + x2(n)
         y2o = y2o + y2(n)
         enddo
 
      x1o = x1o/NUSE
      y1o = y1o/NUSE
      x2o = x2o/NUSE
      y2o = y2o/NUSE
 
      sxx = 0.0
      sx  = 0.0
      syy = 0.0
      sy  = 0.0
      swx = 0.0
      swy = 0.0
      szx = 0.0
      szy = 0.0
      sw  = 0.0
      sz  = 0.0
      sxy = 0.0
      do n = 1, NUSE
         sxy = sxy + (x1(n)-x1o)*(y1(n)-y1o)
         sxx = sxx + (x1(n)-x1o)*(x1(n)-x1o)
         sx  = sx  + (x1(n)-x1o)
         syy = syy + (y1(n)-y1o)*(y1(n)-y1o)
         sy  = sy  + (y1(n)-y1o)
         swx = swx + (x2(n)-x2o)*(x1(n)-x1o)
         swy = swy + (x2(n)-x2o)*(y1(n)-y1o)
         sw  = sw  + (x2(n)-x2o)
         szx = szx + (y2(n)-y2o)*(x1(n)-x1o)
         szy = szy + (y2(n)-y2o)*(y1(n)-y1o)
         sz  = sz  + (y2(n)-y2o)
         enddo
 
 
      dsxx = sx*sx - NUSE*sxx
      dsyy = sy*sy - NUSE*syy
      dsxy = sx*sy - NUSE*sxy
 
      dlta = dsxx*dsyy - dsxy*dsxy
 
      if (dlta.eq.0) goto 999
 
      dswx = sw*sx - NUSE*swx
      dswy = sw*sy - NUSE*swy
      dszx = sz*sx - NUSE*szx
      dszy = sz*sy - NUSE*szy
 
      A =   (dswx*dsyy-dswy*dsxy)/dlta
      B =   (dswy*dsxx-dswx*dsxy)/dlta
      C =   (dszx*dsyy-dszy*dsxy)/dlta
      D =   (dszy*dsxx-dszx*dsxy)/dlta
 
      return
 
 999  continue
      A  = 1
      B  = 0
      C  = 0
      D  = 1
      x1o = 0
      y1o = 0
      x2o = 0
      y2o = 0
      return
      end
 
 
 
 
c-------------------------------------------------
c
c This returns the integer minimum of two numbers
c
      integer function imin(i1,i2)
      implicit none
 
      integer i1, i2
 
      imin = i1
      if (imin.gt.i2) imin = i2
 
      return
      end
 



      
c-----------------------------------
c
c returns the max value for an array
c
      real*8 function maxarr(x,N)
      implicit none
      integer N
      real*8 x(N)
 
      integer i
      maxarr = x(1)
      do i = 2, N
         if (x(i).gt.maxarr) maxarr=x(i)
         enddo
      end
                                              
 
c------------------------------------
c
c returns the minimum value in an array
c
      real*8 function minarr(x,N)
      implicit none
      integer N
      real*8 x(N)
 
      integer i
      minarr = x(1)
      do i = 2, N
         if (x(i).lt.minarr) minarr=x(i)
         enddo
      end
 
c---------------------------------
c
c swaps two integers
c
      subroutine swip(i1,i2)
      implicit none
      integer i1, i2
      integer temp
      temp = i1
      i1 = i2
      i2 = temp
      return
      end
 
 
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/ROUTINES/GEN/swap.f"
c****                                         
c*********************************************
                                              
 
      subroutine swap(x,y)
      implicit none
      real x, y, t
      t = x
      x = y
      y = t
      return
      end
 
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/ROUTINES/GEN/swdp.f"
c****                                         
c*********************************************
                                              
 
c---------------------------------
c
c swaps two double-precision numbers
c
      subroutine swdp(x,y)
      implicit none
      real*8 x, y, t
      t = x
      x = y
      y = t
      return
      end

      
 
c
c
c this copies the contents of list (xi,yi,mi)
c into the list (xo,yo,mo), but does so in order
c of increasing mo; it only does this for the
c brightest NMAX stars.
c
c
 
      subroutine ord_copy(xi,yi,mi,NIN,xo,yo,mo,NMAX)
      implicit none
      real*8  xi(*), yi(*), mi(*)
      integer NIN
      real*8  xo(*), yo(*), mo(*)
      integer NMAX
 
      integer n, k
      real*8  mmin
      integer nmin
 
      do n = 1, NIN
         xo(n) = xi(n)
         yo(n) = yi(n)
         mo(n) = mi(n)
         enddo
 
      do n = 1, NMAX
         nmin = n
         mmin = mo(n)
         do k = n+1,NIN
            if (mo(k).lt.mmin) then
               nmin = k
               mmin = mo(k)
               endif
            enddo
         call swdp(xo(n),xo(nmin))
         call swdp(yo(n),yo(nmin))
         call swdp(mo(n),mo(nmin))
         enddo
 
      return
      end
 
 
 

 
      subroutine barsig(xlist,NTOT,bar,sig,NUSE,siglim)
      implicit none
 
      integer NTOT
      real*8 xlist(NTOT)
      real*8 bar
      real*8 sig
      integer NUSE
      real*8 siglim
 
      integer n
      real*8  bsum, ssum
      integer nsum
      integer NIT
 
      bar = 0.e0
      sig = 9.e9
      do NIT = 1, 5
         bsum = 0.
         ssum = 0.
         nsum = 0.
         do n = 1, NTOT
            if (abs(xlist(n)-bar).le.siglim*sig) then
               bsum = bsum + xlist(n)
               ssum = ssum + abs(xlist(n)-bar)
               nsum = nsum + 1
               endif
            enddo
         bar = bsum / nsum
        !sig = sqrt(ssum/nsum)
         sig = abs(ssum/nsum)
         enddo
      NUSE = nsum
 
      if (NUSE.le.0) then
         bar = 0.000
         sig = 0.999
         endif
 
      return
      end
 
 
c-----------------------------------------------
c
c reports whether a point is inside a convex
c polygon
c
      logical function inside_poly(x,y,xl,yl,Ns)
      implicit none
 
      real*8 x,y
      integer Ns
      real*8 xl(*), yl(*)
 
      integer n
      real*8  cross1
      real*8  crossN
      integer Nsu
      real*8 xlu(99), ylu(99)
 
      if (abs(Ns).gt.99) stop 'inside_poly Ns.gt.99'
 
      Nsu = Ns
      if (Nsu.lt.0) then    ! open poly, need to close
           Nsu = abs(Ns) + 1
           do N = 1, Nsu-1
              xlu(N) = xl(N)
              ylu(N) = yl(N)
              enddo
           xlu(Nsu) = xlu(1)
           ylu(Nsu) = ylu(1)
         else
           do N = 1, Ns
              xlu(N) = xl(N)
              ylu(N) = yl(N)
              enddo
         endif
 
      inside_poly = .true.
      cross1 = (x-xlu(1))*(ylu(2)-ylu(1)) - (y-ylu(1))*(xlu(2)-xlu(1))
      do n = 2, Nsu-1
           crossN = (x-xlu(N))*(ylu(N+1)-ylu(N))
     .          - (y-ylu(N))*(xlu(N+1)-xlu(N))
           if (cross1*crossN.le.0) inside_poly = .false.
           enddo
      return
      end
 
 
 
 
c-------------------------------------------------------------
c
c
c
      subroutine add_fill_poly4(xgl,ygl,pk_cvg,PIXDIM)
      implicit none
 
      real*8    xgl(5), ygl(5)
      integer   PIXDIM
      integer*2 pk_cvg(PIXDIM,PIXDIM)
 
      integer imin, imax
      integer jmin, jmax
      logical inside_poly
 
      real*8  xg, yg
      integer i, j
 
 
      imin = max(min(xgl(1),xgl(2),xgl(3),xgl(4)),000000*1.0d0)
      imax = min(max(xgl(1),xgl(2),xgl(3),xgl(4)),PIXDIM*1.0d0)
      jmin = max(min(ygl(1),ygl(2),ygl(3),ygl(4)),000000*1.0d0)
      jmax = min(max(ygl(1),ygl(2),ygl(3),ygl(4)),PIXDIM*1.0d0)
!      write(*,'(10x,''   imin: '',3i8)') imin,imax,PIXDIM
!      write(*,'(10x,''   jmin: '',3i8)') jmin,jmax,PIXDIM
      do i = imin, imax
      do j = jmin, jmax
         xg = i
         yg = j
         if (inside_poly(xg,yg,xgl,ygl,5)) pk_cvg(i,j) = pk_cvg(i,j) + 1
         enddo
         enddo
 
      return
      end
 
 
 
c-----------------------------------------------------------
c
c
c
      subroutine cvg_add_c(pk_cvg,CHIPNO,PIXDIM,
     .                     AG,BG,CG,DG,Xo,Yo,Wo,Zo)
      implicit none
 
      integer   PIXDIM
      integer*2 pk_cvg(PIXDIM,PIXDIM)
 
      integer CHIPNO
      real*8 AG, BG, CG, DG
      real*8 Xo, Yo
      real*8 Wo, Zo
 
c
c master-gc frame location of chip edges...
c as a function of chip number
c
      common /xy_2pkc_/ xy_2pkc
      integer xy_2pkc(2,4,4,9)
      data    xy_2pkc /
c
c pc1 --> c=1 (not done yet)
c
     .   0000,0000, 0000,0000, 0000,0000, 0000,0000,
     .   0000,0000, 0000,0000, 0000,0000, 0000,0000,
     .   0000,0000, 0000,0000, 0000,0000, 0000,0000,
     .   0000,0000, 0000,0000, 0000,0000, 0000,0000,
c
c wf2 --> c=2 (not done yet)
c
     .   0000,0000, 0000,0000, 0000,0000, 0000,0000,
     .   0000,0000, 0000,0000, 0000,0000, 0000,0000,
     .   0000,0000, 0000,0000, 0000,0000, 0000,0000,
     .   0000,0000, 0000,0000, 0000,0000, 0000,0000,
c
c wf3 --> c=3 (not done yet)
c
     .   0000,0000, 0000,0000, 0000,0000, 0000,0000,
     .   0000,0000, 0000,0000, 0000,0000, 0000,0000,
     .   0000,0000, 0000,0000, 0000,0000, 0000,0000,
     .   0000,0000, 0000,0000, 0000,0000, 0000,0000,
c
c wf4 --> c=4 (not done yet)
c
     .   0000,0000, 0000,0000, 0000,0000, 0000,0000,
     .   0000,0000, 0000,0000, 0000,0000, 0000,0000,
     .   0000,0000, 0000,0000, 0000,0000, 0000,0000,
     .   0000,0000, 0000,0000, 0000,0000, 0000,0000,
c
c acs ---> c=5
c
     .   0040,-140, 4130,0110, 4110,2190, 0060,1885,
     .   0060,1950, 4110,2255, 4090,4260, 0110,3900,
     .   0000,0000, 0000,0000, 0000,0000, 0000,0000,
     .   0000,0000, 0000,0000, 0000,0000, 0000,0000,
c
c acsx --> c=6 (not done)
c
     .   0000,0000, 0000,0000, 0000,0000, 0000,0000,
     .   0000,0000, 0000,0000, 0000,0000, 0000,0000,
     .   0000,0000, 0000,0000, 0000,0000, 0000,0000,
     .   0000,0000, 0000,0000, 0000,0000, 0000,0000,
c
c hrc --> c=7 (not done yet)
c
     .   0007,0019,    3,0901, 1010,1001, 1006, 121,
     .   0000,0000, 0000,0000, 0000,0000, 0000,0000,
     .   0000,0000, 0000,0000, 0000,0000, 0000,0000,
     .   0000,0000, 0000,0000, 0000,0000, 0000,0000,
 
c
c wfc3uvis ---> c=8
c
     .   0010,0080, 4110,0325, 4095,2375, 0023,2108,
     .   0023,2152, 4095,2425, 4080,4450, 0030,4155,
     .   0000,0000, 0000,0000, 0000,0000, 0000,0000,
     .   0000,0000, 0000,0000, 0000,0000, 0000,0000,
c
c wfc3ir ---> c=9
c
     .   -049,0013, 1061,0013, 1017,1018, -059,1018,
     .   0000,0000, 0000,0000, 0000,0000, 0000,0000,
     .   0000,0000, 0000,0000, 0000,0000, 0000,0000,
     .   0000,0000, 0000,0000, 0000,0000, 0000,0000/
 
      real*8 xcl(5), ycl(5)
      real*8 xgl(5), ygl(5)
 
      integer i, j
      integer k, p
 
 
      if (CHIPNO.le.01.or.CHIPNO.ge.10) then
         do i = 0001, PIXDIM
         do j = 0001, PIXDIM
            pk_cvg(i,j) = 1
            enddo
            enddo
         return
         endif
 
 
      do k = 1, 4
         if (xy_2pkc(1,1,k,CHIPNO).ne.0) then
!            print*,' '
!            write(*,'(10x,''      k:'',i1)') k
            do p = 1, 4
               xcl(p) = xy_2pkc(1,p,k,CHIPNO)
               ycl(p) = xy_2pkc(2,p,k,CHIPNO)
               enddo
            xcl(5) = xy_2pkc(1,1,k,CHIPNO)
            ycl(5) = xy_2pkc(2,1,k,CHIPNO)
            do p = 1, 5
               xgl(p) = (AG*(xcl(p)-Xo)+BG*(ycl(p)-Yo) + Wo)
               ygl(p) = (CG*(xcl(p)-Xo)+DG*(ycl(p)-Yo) + Zo)
               enddo
!            do p = 1, 5
!               write(*,'(10x,''      p:'',i1,4i8)')
!     .                   p,int(xgl(p)),int(ygl(p)),
!     .                     int(xcl(p)),int(ygl(p))
!               enddo
            call add_fill_poly4(xgl,ygl,pk_cvg,PIXDIM)
            endif
         enddo
 
      return
      end


      

      
      subroutine glob_fit6nrDPu(x1,y1,x2,y2,u,Ns,Nu,
     .                          A,B,C,D,x1o,y1o,x2o,y2o,nstmax)
      implicit none

      integer nstmax
      real*8 x1(nstmax), y1(nstmax)
      real*8 x2(nstmax), y2(nstmax)
      integer u(nstmax)
      integer Ns, Nu
      real*8 A, B, C, D
      real*8 x1o, y1o, x2o, y2o

      real*8 sxx, sx, swx, szx
      real*8 syy, sy, swy, szy
      real*8 sw,  sz, sxy
      real*8 dlta
      real*8 dsxx, dsyy, dsxy
      real*8 dswx, dswy, dszx, dszy

      real*8 x1t, y1t
      real*8 x2t, y2t

      integer n

      if (Ns.lt.3) goto 999

      x1t = 0
      y1t = 0
      x2t = 0
      y2t = 0
      Nu  = 0
      do n = 1, Ns
         if (u(n).eq.1) then
            x1t = x1t + x1(n)
            y1t = y1t + y1(n)
            x2t = x2t + x2(n)
            y2t = y2t + y2(n)
            Nu  = Nu  + 1
            endif
         enddo

      x1o = x1t/Nu
      y1o = y1t/Nu
      x2o = x2t/Nu
      y2o = y2t/Nu

      sxx = 0.0
      sx  = 0.0
      syy = 0.0
      sy  = 0.0
      swx = 0.0
      swy = 0.0
      szx = 0.0
      szy = 0.0
      sw  = 0.0
      sz  = 0.0
      sxy = 0.0
      do n = 1, Ns
         sxy = sxy + u(n)*(x1(n)-x1o)*(y1(n)-y1o)
         sxx = sxx + u(n)*(x1(n)-x1o)*(x1(n)-x1o)
         sx  = sx  + u(n)*(x1(n)-x1o)
         syy = syy + u(n)*(y1(n)-y1o)*(y1(n)-y1o)
         sy  = sy  + u(n)*(y1(n)-y1o)
         swx = swx + u(n)*(x2(n)-x2o)*(x1(n)-x1o)
         swy = swy + u(n)*(x2(n)-x2o)*(y1(n)-y1o)
         sw  = sw  + u(n)*(x2(n)-x2o)
         szx = szx + u(n)*(y2(n)-y2o)*(x1(n)-x1o)
         szy = szy + u(n)*(y2(n)-y2o)*(y1(n)-y1o)
         sz  = sz  + u(n)*(y2(n)-y2o)
         enddo


      dsxx = sx*sx - Nu*sxx
      dsyy = sy*sy - Nu*syy
      dsxy = sx*sy - Nu*sxy
   
      dlta = dsxx*dsyy - dsxy*dsxy
      if (dlta.eq.0) goto 999

      dswx = sw*sx - Nu*swx
      dswy = sw*sy - Nu*swy
      dszx = sz*sx - Nu*szx
      dszy = sz*sy - Nu*szy

      A =   (dswx*dsyy-dswy*dsxy)/dlta
      B =   (dswy*dsxx-dswx*dsxy)/dlta
      C =   (dszx*dsyy-dszy*dsxy)/dlta
      D =   (dszy*dsxx-dszx*dsxy)/dlta

      return

 999  continue
      A  = 1
      B  = 0
      C  = 0
      D  = 1
      x1o = 0  
      y1o = 0
      x2o = 0
      y2o = 0
      return
      end





      subroutine jedimatch(xr,yr,nr,Ns,
     .                     rg,dg,ng,Gs,
     .                     n1match,n2match,RD_RAD,RA_CEN,DE_CEN)
      implicit none

      character*300 INPUT
      character*20 INSTRUMENT, DETECTOR

      real*8 RD_RAD, RA_CEN, DE_CEN
      integer Ns, N, i, j, k, Gs, G
      real*8 xr(Ns), yr(Ns), u, v, gdu, gdv, ra, dec
      integer nr(Ns)
      real*8 rg(Gs), dg(Gs), drg(Gs), ddg(Gs), dist(Gs)
      integer ng(Gs), naux(Gs)
      integer n1match(_NSTMAX_), n2match(_NSTMAX_)
      real*8 rd2x, rd2y

      integer nchip, ii, jj
      integer NMATCH
      
          
      NMATCH = 0
      do N = 1, Ns

!     Projected HST positions in arcsec
         u = rd2x(xr(N),yr(N),RA_CEN,DE_CEN)*3600.d0
         v = rd2y(xr(N),yr(N),RA_CEN,DE_CEN)*3600.d0

!     Projected Gaia positions in arcsec
         do G = 1, Gs
            drg(G) = rd2x(rg(G),dg(G),RA_CEN,DE_CEN)*3600.d0
            ddg(G) = rd2y(rg(G),dg(G),RA_CEN,DE_CEN)*3600.d0
         enddo

         dist = sqrt((drg-u)**2+(ddg-v)**2)
         naux = ng
         call cosort2di(dist,naux,Gs)
         if (dist(1).le.RD_RAD) then
            NMATCH = NMATCH + 1
            n1match(NMATCH) = nr(N)
            n2match(NMATCH) = naux(1)
         endif
         
      enddo
          
      return

      end

      
