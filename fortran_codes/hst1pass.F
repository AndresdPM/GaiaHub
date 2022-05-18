 
c----------------------------------------------------
c
c things to improve before release
c 1) wfpc2 -- Dana
c 2) hrc (check)
c 3) acs gc (SM3/SM4)
c 4) get the links to the HST webpages to be consistent;
c    (don't use if not needed)
c 5) wfc3ir (how is it dealing with up the ramp?
c 6) "2pass, maybe?" said Jena.
c
c----------------------------------------------------
 
 
c
c make sure it runs without GDC if it isn't needed...
c
 
#define _PI_    3.141592653589793238462643d0
 
c
c need to make sure WFC3/IR subarrays are properly
c placed...
 
 
c
c enter the filenames by list
c
 
c
c the filename for PSFPERT doesn't actually get used; it's always
c treated as 'AUTO'
c
 
c
c somehow the filter and PID and rdate are not getting populated
c
c    jbts01ycq_flc
c
c #   ARG000: ./hst2xym.2015.09.23.e
c #   ARG001: HMIN=7
c #   ARG002: FMIN=10000
c #   ARG003: PMAX=9e9
c #   ARG004: OUT=xymXYMpq
c
c--------------------------------------------------------------
c
c 2017.02.06
c * added capability to read _raw files for ACS
c
c 2016.06.02
c * need to add sky to possible output list...
c
c
c 2015.07.24
c * add option to do artificial stars...
c
c 2015.06.29
c * get it to read in the images and provide ALL of the  needed info
c   at the time:  transformations, boundaries, etc.
c
c 2015.06.12
c * get it to reead in and work on wfc3ir images
c * I'll worry later about the geometric stuff
c
c 2015.05.06
c * turn the main finding part into a subroutine,
c   so that I can put a wrapper around it... scary
c
c 2015.04.18
c * I turned off the normalization of the pertpsf... it was causing problems
c
c things yet to do
c * internalize the boundary thing (set based on chip at the beginning)
c * make sure I don't need the old-fashioned GCs anymore
c
c 2015.03.09c
c * put the new distortion effort into the measured positions
c * output the 2MASS stuff as region file
c
c 2015.03.06
c * compartmentalize/automate the way it does the distortion correction
c * simplifity the way it does the xy2rd mapping
c * simplifify the way it does the rd2uv mapping
c
c------------------------------------------------------------------------
 
 
c------------------------------------------------------------------------
c
c Monday Feb 23, 2015
c
c NEEDED IMPROVEMENTS
c 01) FLEXIBLE OUTPUT FILE uvMrdxyXYspqcnt
c 02) add imin/imax and jmin/jmax (make sure it works for satd, too!)
c 03) output the important aspects right at the top (include detector)
c
c MAJOR RESTRUCTURING
c --> go thru args only once
c
c------------------------------------------------------------------------
 
c
c------------------------------------------------------------------------
c
 
#define _LINUX_  .true.     /* IS THE MACHINE BIG/LITTLE ENDIEN?       */
#define _NITPRT_      9     /* NUM OF ITERATIONS TO GET PRTPSF         */
#define _NPRTMX_      5     /* MAX NUMBER OF PERT ZONES                */
#define _FIELDs_     26     /* NUMBER OF OUTPUT FIELDS TO CHOOSE FROM  */
#define _NIMMXP_    400     /* MAX NUMBER OF IMAGES (FOR PERT)         */
#define _NIMMAX_  99999     /* MAX NUMBER OF IMAGES                    */
#define _OMAX_     9999     /* MAX NUMBER OF 2MASS STARS TO SEARCH FOR */
#define _NLMAX_   19999     /* MAX NUMBER OF STARS FOR PSF PERTURBN    */
#define _NSTMAX_ 999999     /* MAX NUMBER OF STARS IN AN IMAGE         */
#define _ARTMAX_ 999999     /* MAX NUMBER OF STARS IN AN IMAGE         */
#define _SKIPPX_ .false.
 
      program hst2xym
      implicit none
 
      integer TESTING123
 
      character*080 PROGNAME
 
      character*080 PSFFILE_000
      character*080 PSFFILE_INP
      character*080 PSFFILE_LIB
      character*080 PSFPERT_INP
      character*080 PSFFILE_PRT
      character*080 PSFFILE_USE
      character*200 FILENAME
      character*200 FILENAME_N(_NIMMAX_)
      character*080 WCSMODE
      character*080 CATMAT
      character*080 CATREG
      character*080 OUTDIR
      character*080 OUTDIRu
 
      character*080 ARTFILE_INP
 
      character*080 GDCFILE_000
      character*080 GDCFILE_INP
      character*080 GDCFILE_LIB
      character*080 GDCFILE_USE
 
      character*80 SHOW_ALL
      character*80 SHOW_USE
      character*80 SHOW_FND
      character*80 SHOW_REF
      character*80 SHOW_SUB
      character*80 SHOW_MSK
 
      character*80 STRING
 
      integer  i,  j
      integer iu
 
      integer HMIN
      real    PMAX
      real    FMIN
 
      integer iargc, NARG, NARGs
      integer LNC
 
      integer      ISIMG_NARG(999)
      character*80 STRNG_NARG(999)
 
      logical DOAPPHOT
      real    RAP
      integer SKI,SKO
 
      logical DOSATD
      logical DOHSORT
      logical DOMSORT
      logical DOSTARDB
 
      integer HIFLAG
      common / HIFLAG_ / HIFLAG
      data HIFLAG / 0 /
 
      integer LOFLAG
      common / LOFLAG_ / LOFLAG
      data LOFLAG / -0250 /
 
c----------------------------------
c
c here are the 9x9 fiducial PSFs
c
      integer NIM, NIMs
      real    QMAX
      real    CMIN, CMAX
      logical QCAND
 
      integer KSEL
 
      integer      NPERTs
      real         FOCUS_LEVELi
 
      real pertimg(501,1501)  ! can do up to 150 images...
 
      integer H, HHo, HHs
      character*249 HH(999)   ! max 999 header pages!
 
      integer       NLISTs
      character*80  OUTLIST_NL(9)  ! the suffix of the output file
      integer       NITEMSO_NL(9)  ! the number of elements to output
 
      integer       NREGs
      character*80  OUTLIST_NR(9)  ! the suffix of the output file
      integer       NITEMSO_NR(9)  ! the number of elements to output
 
      integer IMIN, IMAX
      integer JMIN, JMAX
 
c
c----------------------------------------------------------------
c
      integer        N, Ns
      real*8         u_n(_NSTMAX_),  v_n(_NSTMAX_), mm_n(_NSTMAX_)
      real*8         x_n(_NSTMAX_),  y_n(_NSTMAX_),  m_n(_NSTMAX_)
      real*8        xx_n(_NSTMAX_), yy_n(_NSTMAX_)
      real*8        uu_n(_NSTMAX_), vv_n(_NSTMAX_)
      integer        k_n(_NSTMAX_)
      integer        h_n(_NSTMAX_)
      real          hh_n(_NSTMAX_)
      real*8         r_n(_NSTMAX_),  d_n(_NSTMAX_)
      integer        i_n(_NSTMAX_),  j_n(_NSTMAX_)
      real           p_n(_NSTMAX_)
      real           q_n(_NSTMAX_),  c_n(_NSTMAX_)
      real           s_n(_NSTMAX_)
 
      real*8 BDRY_XR(4,4), BDRY_YR(4,4)
      real*8 BDRY_XC(4,4), BDRY_YC(4,4)
      real*8 BDRY_UG(4,4), BDRY_VG(4,4)
      real*8 BDRY_RA(4,4), BDRY_DE(4,4)
      real*8 BDRY_UU(4,4), BDRY_VV(4,4)
 
 
c
c----------------------------------------------------------------
c
 
      character*080 DIRECT
      character*080 PREFIX
      character*7   SUFFIX
 
 
      character*80 FILEOUT
 
      character*05 FILT_N(_NIMMAX_)
      real         EXPT_N(_NIMMAX_)
      real         RDAT_N(_NIMMAX_)
      integer      PROP_N(_NIMMAX_)
      integer      INST_N(_NIMMAX_)
 
      real*8  CRPIX1_USE, CRPIX2_USE
      real*8  CRVAL1_USE, CRVAL2_USE
      real*8   CD1_1_USE,  CD1_2_USE
      real*8   CD2_1_USE,  CD2_2_USE
 
      integer       NREG
 
      integer As
      real*8  xinp_a(_ARTMAX_)
      real*8  yinp_a(_ARTMAX_)
      real*8  minp_a(_ARTMAX_)
 
c
c----------------------------------------------------------------
c
 
      do i = 001, 0501
      do j = 001, 1501
         pertimg(i,j) = 0.
         enddo
         enddo
 
      NARGs = iargc()
 
c
c if these aren't changed from their defaults; the routine will complain
c
      HMIN = -99
      FMIN = 999999
 
c-----------------------------------------------------------------------
 
      GDCFILE_000 = 'AUTO'
      GDCFILE_INP = 'AUTO'
      GDCFILE_LIB = 'NONE'
      GDCFILE_USE = 'NONE'
 
      PSFFILE_000 = 'AUTO'
      PSFFILE_INP = 'NONE'
      PSFFILE_LIB = 'NONE'
      PSFFILE_USE = 'NONE'
      PSFPERT_INP = 'NONE'
      PSFFILE_PRT = 'NONE'
 
      DOSATD     = .true.
      DOHSORT    = .false.
      DOMSORT    = .false.
      DOSTARDB   = .false.
 
      NPERTs       = 0
      FOCUS_LEVELi = 0.
 
      As = 0
 
      KSEL       =  0
      QMAX       =  9.99
      CMIN       = -1.00
      CMAX       =  0.10
      QCAND      =  .false.
 
      SHOW_ALL   = 'NO'
      SHOW_USE   = 'NO'
      SHOW_FND   = 'NO'
      SHOW_REF   = 'NO'
      SHOW_MSK   = 'X'
      SHOW_SUB   = 'NO'
 
      WCSMODE = 'AUTO'
      CATMAT  = 'NONE'
      CATREG  = 'NONE'
      NLISTs   = 0
      NREGs    = 0
 
      IMIN = -999
      IMAX = 9999
      JMIN = -999
      JMAX = 9999
 
      OUTDIR = 'NONE'
 
      DOAPPHOT =  .false.
 
      PMAX = 999999
 
      if (iargc().eq.0) then
         print*,'                                          '
         print*,'This routine takes sevaral args (3 req)   '
         print*,'                                          '
         print*,'hst2xym HMIN=5                            '
         print*,'        FMIN=1000                         '
         print*,'        [PSF=AUTO/FILE/APPHOT]            '
         print*,'        [GDC=AUTO/FILE/NONE]  (distortn)  '
         print*,'        [PMAX=99999]                      '
         print*,'        [QMAX= 0.1]                       '
         print*,'        [CMIN=-1.0]                       '
         print*,'        [CMIN= 0.1]                       '
         print*,'        [SHOW_USE=NO/AUTO/X] image search '
         print*,'        [SHOW_FND=NO/AUTO/X] find report  '
         print*,'        [SHOW_REF=NO/AUTO/X] ref-frame img'
         print*,'        [SHOW_SUB=NO/AUTO/X] subt image   '
         print*,'        [SHOW_MSK=NO/AUTO/X] mask         '
         print*,'        [SHOW_ALL=NO/AUTO/X] show all     '
         print*,'        [PERTn=YES/AUTO/FILE]             '
         print*,'        [FOCUS=-1(find),0(mid),rF]        '
         print*,'        [REF-FRAME PARAMETERS]            '
         print*,'        [IMIN=500] [IMAX=510]             '
         print*,'        [JMIN=922] [JMAX=932]             '
         print*,'        [WCS=FIND/1ST]                    '
         print*,'           [HDR=]                         '
         print*,'           [RDC=]                         '
         print*,'        [MSORT+] [HSORT+]                 '
         print*,'        [STARDB+]                         '
         print*,'        [OUT=uvMrd]                       '
         print*,'        [REG=xy,XY,uv,rd]                 '
         print*,'        [CATMAT=2MASS/1st]                '
         print*,'        [CATREG=FILE]                     '
         print*,'        [CAT_UV=FILE.uv...]               '
         print*,'        [CAT_RD=FILE.rd...]               '
         print*,'        [ARTXYM=FILE.xym]                 '
         print*,'        [FITSs=LIST OF FILES, ONE / LINE] '
         print*,'        IMG1.fits IMG2.fits ...           '
         print*,'                                          '
         print*,'"PSF=APPHOT 2.99 6 9"                     '
         print*,'      ---> for aperture photy             '
         print*,'           w/ap (r<2.99) and              '
         print*,'           sky bet 6 & 9                  '
         print*,'                                          '
         print*,'KSEL  -- use only this chip number        '
         print*,'QMAX  -- this is the maximum allow QFIT   '
         print*,'CMIN  -- this is the minimum central exess'
         print*,'         relative to a point source       '
         print*,'         (use to throw out resolved obj)  '
         print*,'CMAX  -- this is the maximum central exess'
         print*,'         relative to a point source       '
         print*,'         (use to throw out CRs)           '
         print*,'QCAND -- requires both q and c together,  '
         print*,'         (not or)                         '
         print*,'                                                   '
         print*,'REFERENCE FRAME PARAMETERS...                      '
         print*,'     OPTION#0:  AUTO (default)                     '
         print*,'     OPTION#1:  "WCS=IMAGE_WCS.txt"                '
         print*,'                 (wcs info in textfile)            '
         print*,'     OPTION#2:  "HDR=IMAGE.fits"                   '
         print*,'                 (wcs info from fits header)       '
         print*,'     OPTION#3:  "RDC=(00:23:13.5,+33:44:13.22)"    '
         print*,'                 (reference center for 50mas frame)'
         print*,'                 (hard format)                     '
         print*,'                                                   '
         print*,'OUT= output file can be made flexibly              '
         print*,'     u,v = ref frame u and v position              '
         print*,'     m   = instrumental mag                        '
         print*,'     M   = instrumental mag (1000s, pix-area corr) '
         print*,'     V   = VEGA MAG (not available yet)            '
         print*,'     x,y = raw chip-dependent x y coord            '
         print*,'     k   = chip number (neg for xy in abtd system) '
         print*,'     X,Y = dist corr x y                           '
         print*,'     h   = isolation index (pixel based)           '
         print*,'     H   = isolation index (output-file based)     '
         print*,'     r,R = Right Ascension (R in hex)              '
         print*,'     r,D = Declination (D in hex)                  '
         print*,'     s   = sky value                               '
         print*,'     p   = brightest pixel value                   '
         print*,'     q   = quality of fit (0 = perfect)            '
         print*,'     c   = chisq of fit                            '
         print*,'     k,K = chip number (cap gives K1)              '
         print*,'                                                   '
         print*,'OUTDIR= specify directory for OUT and REG files... '
         print*,'       OUTDIR=NONE  -- default, put in run dir     '
         print*,'       OUTDIR=IMAGE -- put in image directories    '
         print*,'       OUTDIR=<DIR> -- send to specific directory  '
         print*,'                                                   '
         stop
         endif
 
      do H = 1, 999
         write(HH(H),'(''#'',248('' ''))')
         enddo
 
      call getarg(0,PROGNAME)
 
      write(HH(1),'(''#'')')
      write(HH(2),'(''#--------------------------------------------'')')
      write(HH(3),'(''# ARGUMENTS '')')
      write(HH(4),'(''#--------------------------------------------'')')
      write(HH(5),'(''#   ARG'',i4.4,'': '',a80)') 0,PROGNAME
      HHo = 5
 
c
c----------------------------------------------------------------
c
 
 
 
      print*,' '
      print*,' '
      write(*,'(''ARG'',i4.4,2x,80a)') 0,PROGNAME
      NIMs = 0
      do NARG = 1, NARGs
         ISIMG_NARG(NARG) = 0
         call getarg(NARG,FILENAME)
         HHo = HHo + 1
         write(HH(HHo),'(''#   ARG'',i4.4,'': '',a80)') NARG,FILENAME
         STRNG_NARG(NARG) = FILENAME
         write(*,'(''ARG'',i4.4,2x,80a)') NARG,FILENAME
         if (FILENAME(1:5).eq.'HMIN=') then
            read(FILENAME(6:80),*) HMIN
            goto 555
            endif
         if (FILENAME(1:5).eq.'FMIN=') then
            read(FILENAME(6:80),*) FMIN
            goto 555
            endif
         if (FILENAME(1:5).eq.'PMAX=') then
            read(FILENAME(6:80),*) PMAX
            goto 555
            endif
         if (FILENAME(1:4).eq.'PSF=') then
            PSFFILE_000 = FILENAME(5:80)
            PSFFILE_INP = FILENAME(5:80)
            goto 555
            endif
         if (FILENAME(1:4).eq.'GDC=') then
            GDCFILE_000 = FILENAME(5:80)
            goto 555
            endif
         if (FILENAME(1:7).eq.'LOFLAG=') then
            read(FILENAME(8:20),*) LOFLAG
            goto 555
            endif
         if (FILENAME(1:7).eq.'HIFLAG=') then
            read(FILENAME(8:20),*) HIFLAG
            goto 555
            endif
         if (FILENAME(1:6).eq.'HIAUTO') then
            HIFLAG = 0
            goto 555
            endif
         if (FILENAME(1:4).eq.'WCS=') then
            WCSMODE = FILENAME(5:80)
            goto 555
            endif
         if (FILENAME(1:4).eq.'HDR=') then
            WCSMODE = FILENAME
            goto 555
            endif
         if (FILENAME(1:4).eq.'RDC=') then
            WCSMODE = FILENAME
            goto 555
            endif
         if (FILENAME(1:7).eq.'DOSATD+') then
            DOSATD = .true.
            goto 555
            endif
         if (FILENAME(1:7).eq.'DOSATD-') then
            DOSATD = .false.
            goto 555
            endif
         if (FILENAME(1:9).eq.'SHOW_USE=') then
            SHOW_USE = FILENAME(10:80)
            goto 555
            endif
         if (FILENAME(1:9).eq.'SHOW_FND=') then
            SHOW_FND = FILENAME(10:80)
            goto 555
            endif
         if (FILENAME(1:9).eq.'SHOW_REF=') then
            SHOW_REF=FILENAME(10:80)
            if (SHOW_REF(1:2).ne.'NO'.and.
     .          SHOW_REF(1:4).ne.'AUTO') then
                print*,' SHOW_REF: ',SHOW_REF
                print*,' (can only be NO or AUTO)'
                stop
                endif
            goto 555
            endif
         if (FILENAME(1:9).eq.'SHOW_SUB=') then
            SHOW_SUB=FILENAME(10:80)
            if (SHOW_SUB(1:2).ne.'NO'.and.
     .          SHOW_SUB(1:4).ne.'AUTO') then
                print*,' SHOW_SUB: ',SHOW_SUB
                print*,' (can only be NO or AUTO)'
                stop
                endif
            goto 555
            endif
         if (FILENAME(1:9).eq.'SHOW_MSK=') then
            SHOW_MSK=FILENAME(10:80)
            if (SHOW_MSK(1:1).ne.'X' .and.
     .          SHOW_MSK(1:2).ne.'NO'.and.
     .          SHOW_MSK(1:4).ne.'AUTO') then
                print*,' SHOW_MSK: ',SHOW_MSK
                print*,' (can only be X, NO or AUTO)'
                stop
                endif
            goto 555
            endif
         if (FILENAME(1:6).eq.'FOCUS=') then
            read(FILENAME(7:80),*) FOCUS_LEVELi
            print*,'---> FOCUS_LEVELi: ',FOCUS_LEVELi
            goto 555
            endif
         if (FILENAME(1:4).eq.'PERT') then
            if (FILENAME(5:5).lt.'0'.or.FILENAME(5:5).gt.'9') then
               print*,' '
               print*,'PERTn MUST HAVE AN n VALUE BETWEEN 0 AND 9'
               print*,'ARG: ',FILENAME(1:20)
               print*,' '
               stop
               endif
            read(FILENAME(5:5),*) NPERTs
            print*,'        ---> NPERTs: ',NPERTs
            if (FILENAME(6:6).ne.'=') then
               print*,' '
               print*,'The 6th character of PERT must be = '
               print*,' '
               stop
               endif
            if (NPERTs.gt._NPRTMX_) then
               print*,'hst2xym --- '
               print*,'        --->  NPERTs : ',NPERTs
               print*,'        ---> _NPRTMX_: ',_NPRTMX_
               stop
               endif
            PSFPERT_INP = FILENAME(7:80)
            goto 555
            endif
         if (FILENAME(1:5).eq.'KSEL=') then
            read(FILENAME(6:20),*) KSEL
            goto 555
            endif
         if (FILENAME(1:5).eq.'QMAX=') then
            read(FILENAME(6:20),*) QMAX
            goto 555
            endif
         if (FILENAME(1:5).eq.'CMIN=') then
            read(FILENAME(6:20),*) CMIN
            goto 555
            endif
         if (FILENAME(1:5).eq.'CMAX=') then
            read(FILENAME(6:20),*) CMAX
            goto 555
            endif
         if (FILENAME(1:5).eq.'QCAND') then
            if (FILENAME(6:6).ne.'+'.and.
     .          FILENAME(6:6).ne.'-') then
                print*,'QCAND flag must be + or -...'
                stop
                endif
            if (FILENAME(6:6).eq.'+') QCAND = .true.
            if (FILENAME(6:6).eq.'-') QCAND = .false.
            print*,'---> QCAND: ',QCAND
            goto 555
            endif
         if (FILENAME(1:5).eq.'IMIN=') then
            read(FILENAME(6:80),*) IMIN
            goto 555
            endif
         if (FILENAME(1:5).eq.'IMAX=') then
            read(FILENAME(6:80),*) IMAX
            goto 555
            endif
         if (FILENAME(1:5).eq.'JMIN=') then
            read(FILENAME(6:80),*) JMIN
            goto 555
            endif
         if (FILENAME(1:5).eq.'JMAX=') then
            read(FILENAME(6:80),*) JMAX
            goto 555
            endif
         if (FILENAME(1:6).eq.'MSORT+') then
            DOMSORT = .true.
            goto 555
            endif
         if (FILENAME(1:6).eq.'HSORT+') then
            DOHSORT = .true.
            goto 555
            endif
         if (FILENAME(1:7).eq.'STARDB+') then
            DOSTARDB = .true.
            goto 555
            endif
         if (FILENAME(1:7).eq.'OUTDIR=') then
            OUTDIR = FILENAME(8:80)
            print*,'                   '
            print*,'  Specify OUTDIR: '
            print*,'     OUTDIR = ',OUTDIR
            print*,'                   '
            goto 555
            endif
         if (FILENAME(1:4).eq.'OUT=') then
            NLISTs = NLISTs + 1
            if (NLISTs.gt.9) stop 'NLISTs.gt.9'
            OUTLIST_NL(NLISTs) = FILENAME(5:80)
            NITEMSO_NL(NLISTs) = 0
            do i = 1, 80
               if (OUTLIST_NL(NLISTs)(i:i).ne.' ') NITEMSO_NL(NLISTs)= i
               enddo
            print*,'       --->    NLISTs = ',NLISTs
            print*,'       --->    NITEMs = ',NITEMSO_NL(NLISTs)
            print*,'       --->   OUTLIST = ',
     .                        OUTLIST_NL(NLISTs)(1:NITEMSO_NL(NLISTs))
            goto 555
            endif
         if (FILENAME(1:4).eq.'REG=') then
            NREGs = NREGs + 1
            if (NREGs.gt.9) stop 'NREGs.gt.9'
            OUTLIST_NR(NREGs) = FILENAME(5:80)
            NITEMSO_NR(NREGs) = 0
            do i = 1, 80
               if (OUTLIST_NR(NREGs)(i:i).ne.' ') NITEMSO_NR(NREGs)= i
               enddo
            print*,'       --->     NREGs = ',NREGs
            print*,'       --->    NITEMs = ',NITEMSO_NR(NREGs)
            print*,'       --->   OUTLIST = ',
     .                     OUTLIST_NR(NREGs)(1:NITEMSO_NR(NREGs))
            if (NITEMSO_NR(NREGs).ne.2)
     .          stop 'can only have two positions in reg file'
            if (OUTLIST_NR(NREGs)(1:2).ne.'xy'.and.
     .          OUTLIST_NR(NREGs)(1:2).ne.'XY'.and.
     .          OUTLIST_NR(NREGs)(1:2).ne.'uv'.and.
     .          OUTLIST_NR(NREGs)(1:2).ne.'rd')
     .              stop 'CAN ONLY MAKE REGION OF xy XY uv rd'
            goto 555
            endif
         if (FILENAME(1:7).eq.'CATMAT=') then
            CATMAT = FILENAME(8:80)
            goto 555
            endif
         if (FILENAME(1:7).eq.'CATREG=') then
            CATREG = FILENAME(8:80)
            goto 555
            endif
         if (FILENAME(1:7).eq.'ARTXYM=') then
            ARTFILE_INP = FILENAME(8:80)
            open(19,file=ARTFILE_INP,status='old')
            As = 0
    1       read(19,'(a80)',end=2) STRING
            if (STRING(1:1).eq.'#') goto 1
            As = As + 1
            read(STRING,*) xinp_a(As), yinp_a(As), minp_a(As)
            goto 1
    2       continue
            print*,'READ IN --- As: ',As
            close(19)
            goto 555
            endif
         if (FILENAME(1:6).eq.'FITSs=') then
            print*,'OPEN: ',FILENAME(7:80)
            open(19,file=FILENAME(7:80),status='old')
  888       read(19,'(a80)',end=889) STRING
            if (STRING(1:1).eq.'#') goto 888
            iu = 0
            do i = 1, 75
               if (STRING(i:i+4).eq.'.fits') iu = i+4
               enddo
            if (iu.eq.0) stop 'FITSs FILENAME IN LIST MUST HAVE .fits'
            NIMs = NIMs + 1
            if (NIMs.gt._NIMMAX_) then
                print*,'     NIMs : ',NIMs
                print*,'  _NIMMAX_: ',_NIMMAX_
                stop 'NIMs.gt._NIMMAX_'
                endif
            FILENAME_N(NIMs) = STRING(1:iu+4)
            write(*,'(''        ---> NIM'',i4.4,1x,80a)')
     .                               NIMs,FILENAME_N(NIMs)
            goto 888
  889       continue
            print*,'---> NIMs: ',NIMs
            goto 555
            endif
 
c
c at this point, we should only have FITS images left...
c
         iu = 0
         do i = 1, 75
            if (FILENAME(i:i+4).eq.'.fits') iu = i
            enddo
         if (iu.eq.0) stop 'UNRECOGNIZED PARAMETER'
 
         NIMs = NIMs + 1
         HHo  = HHo  - 1
         if (NIMs.gt._NIMMAX_) stop 'NIMs.gt._NIMMAX_'
         FILENAME_N(NIMs) = FILENAME(1:LNC(FILENAME,200))
         write(*,'(''        ---> NIM'',i4.4,1x,80a)') NIMs,FILENAME
  555    continue
         enddo
 
 
 
c----------------------------------------------------------
c
c everything has been read in now from the command line,
c check to make sure that we have enough information to do
c what needs to be done...
c
      if (HMIN.eq.-99) then
         print*,'--> COMMAND LINE MUST INCLUDE HMIN'
         stop
         endif
      if (FMIN.eq.-99) then
         print*,'--> COMMAND LINE MUST INCLUDE FMIN'
         stop
         endif
      if (PSFFILE_INP(1:4).eq.'NULL') then
         print*,'--> COMMAND LINE MUST INCLUDE PSF'
         stop
         endif
      if (NIMs.eq.0) then
         print*,'--> COMMAND LINE MUST INCLUDE AT LEAST ONE IMAGE'
         stop
         endif
 
      print*,' '
      print*,' '
      write(*,'(''            '')')
      write(*,'(''            '')')
      write(*,'(''   OUTPUT FROM PROGRAM hst2xym  '')')
      write(*,'(''            '')')
      write(*,'(''            '',a80)')  PROGNAME
      write(*,'(''            '')')
      write(*,'(''     HMIN: '',i3  )')  HMIN
      write(*,'(''     FMIN: '',f19.1)') FMIN
      write(*,'(''     PMAX: '',f19.1)') PMAX
      write(*,'('' PSFFILE0: '',a80)')   PSFFILE_000
      write(*,'('' GDCFILE0: '',a80)')   GDCFILE_000
      write(*,'('' PSFFILEI: '',a80)')   PSFFILE_INP
      write(*,'('' GDCFILEI: '',a80)')   GDCFILE_INP
      write(*,'('' DOAPPHOT: '',l1)')    DOAPPHOT
      write(*,'(''          '')')
      write(*,'(''   DOSATD: '',l01)') DOSATD
      write(*,'(''  GDC_INP: '',80a)') GDCFILE_INP
      write(*,'(''  GDC_LIB: '',80a)') GDCFILE_LIB
      write(*,'(''  GDC_USE: '',80a)') GDCFILE_USE
      write(*,'(''  PSF_INP: '',80a)') PSFFILE_INP
      write(*,'(''  PSF_LIB: '',80a)') PSFFILE_LIB
      write(*,'('' PERT_INP: '',80a)') PSFPERT_INP
      write(*,'(''   NPERTs: '',i1)')  NPERTs
      write(*,'(''  WCSMODE: '',80a)') WCSMODE
      write(*,'(''     KSEL: '',i2  )') KSEL
      write(*,'(''     QMAX: '',f6.3)') QMAX
      write(*,'(''     CMIN: '',f6.3)') CMIN
      write(*,'(''     CMAX: '',f6.3)') CMAX
      write(*,'(''    QCAND: '',l1)')   QCAND
      write(*,'(''           '')')
      write(*,'(''     NIMs: '',i5)') NIMs
      do NIM = 1, NIMs
         print*,'NIM: ',NIM,FILENAME_N(NIM)(1:60)
         enddo
      write(*,'(''           '')')
 
      if (PSFFILE_INP(1:6).eq.'APPHOT') then
          write(*,'(''#PSFFILEI: '',80a)') PSFFILE_INP
          read(PSFFILE_INP(7:80),*) RAP,SKI,SKO
          write(*,'(''# ---> APERTURE RAP: '',f8.4)') RAP
          write(*,'(''# --->    INNER SKI: '',i3)') SKI
          write(*,'(''# --->    OUTER SKO: '',i3)') SKO
          DOAPPHOT = .true.
          endif
 
      do NIM = 1, NIMs
         GDCFILE_INP = GDCFILE_000
         PSFFILE_INP = PSFFILE_000
 
         call sub_hst2xym(HH,HHo,HHs,
     .                    HMIN,FMIN,PMAX,
     .                    HIFLAG,WCSMODE,DOSATD,
     .                    SHOW_ALL,SHOW_USE,SHOW_FND,
     .                    SHOW_REF,SHOW_SUB,SHOW_MSK,
     .                    NPERTs,
     .                    KSEL,QMAX,CMIN,CMAX,QCAND,
     .                    IMIN,IMAX,JMIN,JMAX,DOMSORT,DOHSORT,
     .                    CATMAT, CATREG,
     .                    FILENAME_N(NIM),NIM,NIMs,
     .                       PSFFILE_INP,
     .                       GDCFILE_INP,
     .                    Ns,  u_n,  v_n, mm_n,
     .                         x_n,  y_n,  m_n,  k_n,
     .                        xx_n, yy_n,
     .                        uu_n, vv_n,
     .                         h_n, hh_n,
     .                         r_n,  d_n,
     .                         i_n,  j_n,  p_n,
     .                         q_n,  c_n,  s_n,
     .                    BDRY_XR, BDRY_YR,
     .                    BDRY_XC, BDRY_YC,
     .                    BDRY_UG, BDRY_VG,
     .                    BDRY_RA, BDRY_DE,
     .                    BDRY_UU, BDRY_VV,
     .                    FILT_N(NIM),EXPT_N(NIM),
     .                    RDAT_N(NIM),PROP_N(NIM),INST_N(NIM),
     .                    CRPIX1_USE, CRPIX2_USE,
     .                    CRVAL1_USE, CRVAL2_USE,
     .                    CD1_1_USE,  CD1_2_USE,
     .                    CD2_1_USE,  CD2_2_USE,
     .                    2,                   ! VERBOSE
     .                    As,xinp_a,yinp_a,minp_a,
     .                    1,FOCUS_LEVELi,DOSTARDB)
 
 
         call dirstrip(FILENAME_N(NIM),DIRECT,PREFIX,SUFFIX)
 
         OUTDIRu = OUTDIR
         if (OUTDIR(1:4).eq.'NONE' ) OUTDIRu = './'
         if (OUTDIR(1:5).eq.'IMAGE') OUTDIRu = DIRECT
 
         if (NLISTs.ne.0)
     .   call output_lists(NLISTs,NITEMSO_NL,OUTLIST_NL,
     .                     OUTDIRu,PREFIX,
     .                     HH, HHs,
     .                     Ns,  u_n,  v_n, mm_n,
     .                          x_n,  y_n,  m_n,  k_n,
     .                         xx_n, yy_n,
     .                         uu_n, vv_n,
     .                          h_n, hh_n,
     .                          r_n,  d_n,
     .                          i_n,  j_n,  p_n,
     .                          q_n,  c_n,  s_n)
 
 
         if (NREGs.ne.0)
     .   call output_regs(NREGs,NITEMSO_NR,OUTLIST_NR,
     .                    OUTDIRu,PREFIX,HIFLAG,
     .                    HH, HHs,
     .                    Ns,  u_n,  v_n, mm_n,
     .                         x_n,  y_n,  m_n,  k_n,
     .                        xx_n, yy_n,
     .                        uu_n, vv_n,
     .                         h_n, hh_n,
     .                         r_n,  d_n,
     .                         i_n,  j_n,  p_n,
     .                         q_n,  c_n)
 
 
 
         enddo ! NIM = 1, NIMs
 
 
      stop
      end
 
 
 
 
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/ROUTINES/GEN/oiojor.f"
c****                                         
c*********************************************
                                              
 
c--------------------------------------------------
c
c This routine orders the pixels by successive distance
c from each other.  If you step through the array
c like this:
c
c
c     do m = 1, 99
c        i = ixc + oi(m)
c        j = iyc + oj(m)
c        ...
c        enddo
c
c
c Then, the pixels (i,j) will be in order of increasing
c distance from the central pixel (ixc,iyc).
c
c This can be useful for many applications.
c
c
      integer function oi(m)
      implicit none
      integer m
 
      integer qm, fm
      integer im
      integer mm
      integer oim, ojm
      integer nm
 
      integer oil(91)
      data oil / 0,1,1,2,2,1,2,3,3,1,3,2,4,4,
     .               1,3,4,2,3,5,4,5,1,5,2,4,
     .           3,5,6,1,6,6,2,5,4,3,6,7,5,1,
     .               7,6,4,2,7,7,3,6,5,8,1,4,
     .           8,7,8,2,6,8,3,7,5,4,8,9,1,9,
     .                 9,7,2,6,5,8,3,9,4,9,7,
     .           9,5,8,6,8,7,8,9,6,9,7,9,8,9 /
 
      integer ojl(91)
      data ojl / 0,0,1,0,1,2,2,0,1,3,2,3,0,1,4,
     .                   3,2,4,4,0,3,1,5,2,5,4,
     .           5,3,0,6,1,2,6,4,5,6,3,0,5,7,1,
     .                   4,6,7,2,3,7,5,6,0,8,7,
     .           1,4,2,8,6,3,8,5,7,8,4,0,9,1,2,
     .                     6,9,7,8,5,9,3,9,4,7,
     .             5,9,6,8,7,8,8,6,9,7,9,8,9,9 /
 
 
      qm = m+2 - 4*int((m+2)/4)
      fm = int((m+6)/4)
      if (fm.le.91) then
         if (qm.eq.0) oi =  oil(fm)
         if (qm.eq.1) oi = -oil(fm)
         if (qm.eq.2) oi = -ojl(fm)
         if (qm.eq.3) oi =  ojl(fm)
         return
         endif
 
      im = (sqrt(1.*m-1)-1)/2
      mm = (2*im+1)**2+1
      nm =  (m-mm)/4
      oim = nm-im
      ojm = im
 
      if (qm.eq.0) oi =  oim
      if (qm.eq.1) oi = -oim
      if (qm.eq.2) oi = -ojm
      if (qm.eq.3) oi =  ojm
 
      return
      end
 
c----------------------------------------------------------------
c
c the y-analog to oi(m)
c
      integer function oj(m)
      implicit none
      integer m
 
      integer qm, fm
      integer im, mm, nm, oim, ojm
 
      integer oil(91)
      data oil / 0,1,1,2,2,1,2,3,3,1,3,2,4,4,
     .               1,3,4,2,3,5,4,5,1,5,2,4,
     .           3,5,6,1,6,6,2,5,4,3,6,7,5,1,
     .               7,6,4,2,7,7,3,6,5,8,1,4,
     .           8,7,8,2,6,8,3,7,5,4,8,9,1,9,
     .                 9,7,2,6,5,8,3,9,4,9,7,
     .           9,5,8,6,8,7,8,9,6,9,7,9,8,9 /
 
      integer ojl(91)
      data ojl / 0,0,1,0,1,2,2,0,1,3,2,3,0,1,4,
     .                   3,2,4,4,0,3,1,5,2,5,4,
     .           5,3,0,6,1,2,6,4,5,6,3,0,5,7,1,
     .                   4,6,7,2,3,7,5,6,0,8,7,
     .           1,4,2,8,6,3,8,5,7,8,4,0,9,1,2,
     .                     6,9,7,8,5,9,3,9,4,7,
     .             5,9,6,8,7,8,8,6,9,7,9,8,9,9 /
 
      qm = m+2 - 4*int((m+2)/4)
      fm = int((m+6)/4)
      if (fm.le.91) then
         if (qm.eq.0) oj =  ojl(fm)
         if (qm.eq.1) oj = -ojl(fm)
         if (qm.eq.2) oj =  oil(fm)
         if (qm.eq.3) oj = -oil(fm)
         return
         endif
 
      im = (sqrt(1.*m-1)-1)/2
      mm = (2*im+1)**2+1
      nm =  (m-mm)/4
      oim = nm-im
      ojm = im
 
      if (qm.eq.0) oj =  ojm
      if (qm.eq.1) oj = -ojm
      if (qm.eq.2) oj =  oim
      if (qm.eq.3) oj = -oim
 
      return
      end
 
c----------------------------------------------------------------
c
c should be obvious
c
      real function orm(m)
      implicit none
      integer m
 
      integer oi, oj
 
      orm = sqrt(1.*(oi(m)**2 + oj(m)**2))
 
      return
      end
 
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/ROUTINES/IMAGE/fnd_hloc_NAXIS.f"
c****                                         
c*********************************************
                                              
 
 
c---------------------------------------------
c
c given a pixel (i,j), how far out in radius do
c you have to go to find a brighter pixel?
c
      integer function fnd_hloc_NAXIS(i,j,pix,NAXIS1,NAXIS2)
      implicit none
 
      integer i, j
      integer NAXIS1
      integer NAXIS2
      real pix(NAXIS1,NAXIS2)
 
      integer m, h
      integer ooi, ooj
      integer  oi,  oj
      real    pcen
      integer ii, jj
 
      pcen = pix(i,j)
      do m = 2, 251
         ooi = oi(m)
         ooj = oj(m)
         h = sqrt((ooi**2+ooj**2)*1.0)
         ii = i + ooi
         jj = j + ooj
         if (ii.lt. 0001 ) goto 1
         if (jj.lt. 0001 ) goto 1
         if (ii.gt.NAXIS1) goto 1
         if (jj.gt.NAXIS2) goto 1
         if (pix(ii,jj).gt.pix(i,j)) goto 1
         enddo
 
 1    continue
      fnd_hloc_NAXIS = h
      return
      end
 
 
 
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/ROUTINES/IMAGE/fnd_ploc_NAXIS.f"
c****                                         
c*********************************************
                                              
 
 
      integer function fnd_ploc_NAXIS(i,j,pix,SKY,NAXIS1,NAXIS2)
      implicit none
 
      integer i, j
      integer NAXIS1
      integer NAXIS2
      real pix(NAXIS1,NAXIS2)
      real    SKY
 
      integer m, h
      integer ooi, ooj
      integer  oi,  oj
      real    pcen
      integer ii, jj
 
      pcen = pix(i,j)
      do m = 2, 251
         ooi = oi(m)
         ooj = oj(m)
         h = sqrt((ooi**2+ooj**2)*1.0)
         ii = i + ooi
         jj = j + ooj
         if (ii.lt. 0001 ) goto 1
         if (jj.lt. 0001 ) goto 1
         if (ii.gt.NAXIS1) goto 1
         if (jj.gt.NAXIS2) goto 1
         if (pix(ii,jj).gt.pix(i,j)) goto 1
         if (m.gt.09.and.pix(ii,jj)-SKY.gt.0.50*(pcen-SKY)) goto 1
         if (m.gt.21.and.pix(ii,jj)-SKY.gt.0.25*(pcen-SKY)) goto 1
         if (m.gt.37.and.pix(ii,jj)-SKY.gt.0.10*(pcen-SKY)) goto 1
         if (m.gt.45.and.pix(ii,jj)-SKY.gt.0.05*(pcen-SKY)) goto 1
         enddo
 
 1    continue
      fnd_ploc_NAXIS = h
      return
      end
 
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/ROUTINES/SORT/rbubble.f"
c****                                         
c*********************************************
                                              
 
c--------------------------------------
c
c bubble-sorts a real*4 list into ascending order
c
      subroutine rbubble(r1,NTOT)
      implicit none
      real r1(*)
      real temp
      integer NTOT
      integer n
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
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/ROUTINES/STAT/histmode_NAXIS.f"
c****                                         
c*********************************************
                                              
 
c-------------------------------------------------
c
c this routine uses a histogram-method to find the mode
c of a large number of pixels, those within (i1:i2,j1:j2)
c of the array pixarr (only use pixels with values that
c are between -10 and 4990)
c
      real function histmode_NAXIS(pixarr,NAXIS1,NAXIS2)
      implicit none
 
      integer NAXIS1, NAXIS2
      real    pixarr(NAXIS1,NAXIS2)
 
      integer i, j, h
      integer mhi, mlo
      real hist_summ
 
      integer hist(5000)
      integer hcum(5000)
      integer NTOT
      real root
 
      real    pcum(101)
      integer ipc
      integer imin
      real    dmin
      real    ptot
 
      integer HIFLAG
      common /HIFLAG_/HIFLAG
      integer LOFLAG
      common /LOFLAG_/LOFLAG
 
 
      if (HIFLAG.lt.  0) stop 'histmode HIFLAG.lt.000'
      if (LOFLAG.gt.100) stop 'histmode LOFLAG.lt.100'
 
      do h = 1,5000
         hist(h) = 0
         hcum(h) = 0
         enddo
 
      NTOT = 0
      PTOT = 0
      do i = 00001+5, NAXIS1-5
      do j = 00001+5, NAXIS2-5
        !if (pixarr(i,j).ge.HIFLAG) goto 2
        !if (pixarr(i,j).le.LOFLAG) goto 2
         h = pixarr(i,j)+10.5
         if (h.lt.   1) goto 2
         if (h.gt.5000) goto 2
         hist(h) = hist(h) + 1
         NTOT = NTOT + 1
         PTOT = PTOT + pixarr(i,j)
 2       continue
         enddo
         enddo
 
      if (NTOT.eq.0) then
          histmode_NAXIS = 0.00
          return
          endif
 
      hcum(1) = hist(1)/2
      do h = 2, 5000
         hcum(h) = hcum(h-1) + (hist(h)+hist(h-1))/2
         ipc = 100.*hcum(h)/NTOT + 1.5
         do i = ipc, 101
            pcum(i) = h
            enddo
         enddo
      do ipc = 1, 101  ! goes from 0% to 100%
         pcum(ipc) = root(NTOT*(ipc-1)/100.,hcum,5000)
         enddo
 
      dmin = pcum(40+1)-pcum(1)
      imin = 1
      do i = 2, 101-40
         if (pcum(40+i)-pcum(i).lt.dmin) then
            dmin = pcum(40+i)-pcum(i)
            imin = i
            endif
         enddo
 
      mhi = pcum(40+imin)+3
      mlo = pcum(imin)   -3
c     print*,'--->  MLO ',mlo
c     print*,'--->  MHI ',mhi
 
      histmode_NAXIS = 0.
      hist_summ = 0.
      do h = mlo, mhi
         hist_summ = hist_summ + hist(h)
         histmode_NAXIS = histmode_NAXIS + hist(h)*(h-10)
         enddo
 
      histmode_NAXIS = histmode_NAXIS / hist_summ
c     print*,'---> HISTMODE: ',histmode_NAXIS
 
      if (hist_summ.le.0) histmode_NAXIS = PTOT / NTOT
c     print*,'--->    MLO-10: ',MLO-10
c     print*,'--->    MHI-10: ',MHI-10
 
      NTOT = 0
      PTOT = 0
      do i = 00001+5, NAXIS1-5
      do j = 00001+5, NAXIS2-5
         if (pixarr(i,j).ge.(mlo-10).and.
     .       pixarr(i,j).le.(mhi-10)) then
             NTOT = NTOT + 1
             PTOT = PTOT + pixarr(i,j)
             endif
         enddo
         enddo
      histmode_NAXIS = PTOT/NTOT
 
c     print*,'         PTOT: ',PTOT
c     print*,'         NTOT: ',NTOT
c     print*,'---> HISTMODE: ',histmode_NAXIS
 
      return
 
      end
 
 
c--------------------------------------
c
c return the value of n where list(n) = v ;
c list must be monotonic increasing
c
      real function root(v,list,NTOT)
      implicit none
      real    v
      integer list(1)
      integer NTOT
      integer nl
 
      do nl = 1, NTOT-2
         if (v.lt.list(nl+1)) goto 3
         enddo
 3    continue
 
      root = nl
      if (list(nl).ne.list(nl+1))
     .    root = nl + 1.*(v-list(nl))/(list(nl+1)-list(nl))
 
      return
      end
 
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/ROUTINES/STRING/dirstrip.f"
c****                                         
c*********************************************
                                              
 
c---------------------------------------------------
c
c takes a filename and splits it off into directory,
c prefix, and suffix
c
      subroutine dirstrip(FULL,DIRECT,PREFIX,SUFFIX)
      implicit none
 
      character*80 FULL
      character*80 DIRECT
      character*80 PREFIX
      character*7  SUFFIX
 
      integer i
 
      integer ndir
      integer ndot
      integer nbrk
      integer nend
 
      ndir = 0
      ndot = 0
      nbrk = 0
      nend = 0
      do i = 1,80
         if (FULL(i:i).eq.'/')  ndir = i
         if (FULL(i:i).eq.'.')  ndot = i
         if (FULL(i:i).eq.'[')  nbrk = i
         if (FULL(i:i).eq.' '.and.
     .       nend.eq.0)  nend = i-1
         enddo
 
      if (ndot.eq.0) then
         print*,' '
         print*,'DIRSTRIP: '
         print*,' '
         print*,'THE IMAGE MUST HAVE A DOT IN ITS NAME.'
         print*,'NAME: ',FULL
         print*,' '
         stop
         endif
 
 
      DIRECT = FULL(1:ndir)
      PREFIX = FULL(ndir+1:ndot-1)
      SUFFIX = FULL(ndot+1:nend  )
 
c     print*,'FULL: ',FULL
c     print*,' ndir: ',ndir,DIRECT
c     print*,' ndot: ',ndot,DIRECT
c     print*,' nend: ',nend,SUFFIX
 
      return
      end
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/ROUTINES/GEN/rclip.f"
c****                                         
c*********************************************
                                              
 
c---------------------------------------
c
c Returns rval if rval is between rlo and rhi,
c otherwise it returns the closer endpoint.  It
c also make sure rval is not a NaN
c
      real function rclip(rval,rlo,rhi)
      implicit none
 
      real rval
      real rlo
      real rhi
 
      rclip = rval
      if (rclip.gt.rhi) rclip = rhi
      if (rclip.lt.rlo) rclip = rlo
      if (.not.(rclip.lt.rhi).and.
     .    .not.(rclip.gt.rlo)) rclip = rhi
 
      return
      end
 
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/ROUTINES/GEN/dclip.f"
c****                                         
c*********************************************
                                              
 
c-------------------------------------------
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
 
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/ROUTINES/PHOT/mbar_sky_NAXIS.f"
c****                                         
c*********************************************
                                              
 
c---------------------------------------------------
c
c This routine will find an iteratively clipped mean sky
c value from an annulis about a point.  The point is (ixc,iyc).
c It will take the pixels between r=inr and r=ior and will
c analyze them for a robust average.
c
      real function mbar_sky_NAXIS(ixc,iyc,inr,ior,
     .                             pixarr,NAXIS1,NAXIS2)
      implicit none
 
      integer ixc, iyc
      integer inr, ior
      integer NAXIS1
      integer NAXIS2
      real pixarr(NAXIS1,NAXIS2)
 
      integer i  , j
      integer ixh, iyh
      real    rij
      integer nuse
 
      real sklist(999999)
      integer nsk
      real bar, sig
 
      integer HIFLAG
      common / HIFLAG_ / HIFLAG
 
      integer LOFLAG
      common / LOFLAG_ / LOFLAG
 
      if (LOFLAG.ge.-1.or.
     .    HIFLAG.le. 1) then
          print*,'---> LOFLAG: ',LOFLAG
          print*,'---> HIFLAG: ',HIFLAG
          stop 'mbar_sky_NAXIS HIFLAG common block undef'
          endif
 
      nsk = 0
      do i = -ior+1, ior-1
      do j = -ior+1, ior-1
         rij = i**2+j**2
         if (rij.ge.ior**2) goto 333
         if (rij.lt.inr**2) goto 333
         ixh = ixc + i
         iyh = iyc + j
         if (ixh.lt. 0001 ) goto 333
         if (iyh.lt. 0001 ) goto 333
         if (ixh.gt.NAXIS1) goto 333
         if (iyh.gt.NAXIS2) goto 333
c        if (pixarr(ixh,iyh).le.LOFLAG) goto 333
c        if (pixarr(ixh,iyh).ge.HIFLAG) goto 333
         if (.not.(pixarr(ixh,iyh).ge.  -100)) goto 333
         if (.not.(pixarr(ixh,iyh).le.  1000)) goto 333
         nsk = nsk + 1
         if (nsk.gt.999999) goto 333
         sklist(nsk) = pixarr(ixh,iyh)
 333     continue
         enddo
         enddo
 
      call rbarsigs(sklist,nsk,bar,sig,nuse,2.50)
 
      mbar_sky_NAXIS = bar
 
      return
      end
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/ROUTINES/WFC.09x10/mbar_skyopt_NAXIS.f"
c****                                         
c*********************************************
                                              
 
 
c-----------------------------------------
c
c this will measure the sky *and* remove the
c star profile before it does; this way it
c can measure a sky near a star which still
c may be very slightly affected by the star
c itself
c
      real function mbar_skyopt_NAXIS(xc,yc,pix,psfloc,
     .                                NAXIS1,NAXIS2)
      implicit none
 
      real*8  xc, yc
      integer NAXIS1
      integer NAXIS2
      real pix(NAXIS1,NAXIS2)
      real psfloc(101,101)
 
      real bar1, sig1
 
      integer ixc, iyc
      integer i, j
      real dx, dy
 
      real FCEN, PCEN
      real ZZZ
      common/zzz/zzz
      real SKY
      real PIXSUM
      real PSFSUM
      integer NNNSUM
      integer n1, n1u
      integer n2
      real sl1(1000)
      real rpsf_phot
      real reff
      common /reff/reff
      integer irmin, irmax, irnxt
      common /SKYINFO_/irmin, irmax
 
 
      integer HIFLAG
      common / HIFLAG_ / HIFLAG
 
      integer LOFLAG
      common / LOFLAG_ / LOFLAG
 
      ixc = int(xc+0.5)
      iyc = int(yc+0.5)
 
      PCEN = 0.
      FCEN = 0.
      do i = -2, 2
      do j = -2, 2
         dx = ixc+i-xc
         dy = iyc+j-yc
         PCEN = PCEN + rpsf_phot(dx,dy,psfloc)
         FCEN = FCEN + pix(ixc+i,iyc+j)
         enddo
         enddo
 
c      print*,' '
c      print*,' IXC: ',IXC
c      print*,' IYC: ',IYC
c      print*,' '
c      print*,'pCEN: ',pix(ixc,iyc)
c      print*,'FCEN: ',FCEN
c      print*,'PCEN: ',PCEN
c      print*,'ZCEN: ',FCEN/PCEN
c      print*,' '
c      print*,' '
 
      irmin = 2
 5    continue
      irmax = irmin+1
      PIXSUM = 0.
      PSFSUM = 0.
      NNNSUM = 0
      do i = -irmax, irmax
      do j = -irmax, irmax
         dx = ixc+i-xc
         dy = iyc+j-yc
         if (i**2+j**2.ge.(irmin-0.5)**2.and.
     .       i**2+j**2.lt.(irmax+0.5)**2.and.
     .       ixc+i.ge.0001.and.ixc+i.le.NAXIS1.and.
     .       iyc+j.ge.0001.and.iyc+j.le.NAXIS1) then
             PIXSUM = PIXSUM + pix(ixc+i,iyc+j)
             PSFSUM = PSFSUM + rpsf_phot(dx,dy,psfloc)
             NNNSUM = NNNSUM + 1
             endif
         enddo
         enddo
 
      ZZZ = 0.0
      SKY = (PIXSUM-ZZZ*PSFSUM)/NNNSUM
      ZZZ = (FCEN-25*SKY)/PCEN
      if (ZZZ.lt.0) ZZZ = 0.
      SKY = (PIXSUM-ZZZ*PSFSUM)/NNNSUM
      ZZZ = (FCEN-25*SKY)/PCEN
      if (ZZZ.lt.0) ZZZ = 0.
      SKY = (PIXSUM-ZZZ*PSFSUM)/NNNSUM
      ZZZ = (FCEN-25*SKY)/PCEN
      if (ZZZ.lt.0) ZZZ = 0.
      SKY = (PIXSUM-ZZZ*PSFSUM)/NNNSUM
      ZZZ = (FCEN-25*SKY)/PCEN
      if (ZZZ.lt.0) ZZZ = 0.
 
      if (PSFSUM*ZZZ/NNNSUM.gt.0.03*(PIXSUM/NNNSUM+15).and.
     .   irmin.lt.16) then
         irmin = irmin + 1
         goto 5
         endif
 
 
 
c-------------------------------------------------
c
c ok, now we have the inner radius; where
c the star contribution is less than 10% of the
c sky brightness
c
      irmin = irmin
      irmax = irmin+3
  6   continue
      irnxt = irmax+1
      n1 = 0
      n2 = 0
      reff = 0.
      do i = -irmax-1, irmax+1
      do j = -irmax-1, irmax+1
         dx = ixc+i-xc
         dy = iyc+j-yc
         if (i**2+j**2.ge.(irmin-0.5)**2.and.
     .       i**2+j**2.lt.(irmax+0.5)**2.and.
     .       ixc+i.ge.0001.and.ixc+i.le.NAXIS1.and.
     .       iyc+j.ge.0001.and.iyc+j.le.NAXIS2) then
                if (n1.lt.1000) n1 = n1 + 1
                sl1(n1) = pix(ixc+i,iyc+j)
     .                   - ZZZ*rpsf_phot(dx,dy,psfloc)
                reff = reff + sqrt(dx**2+dy**2)
                endif
         enddo
         enddo
 
      reff = reff/n1
 
      call barsiggg(sl1,n1,bar1,sig1,n1u)
 
      mbar_skyopt_NAXIS = bar1
 
      return
      end
 
 
c------------------------------------------------
c
c
      subroutine barsiggg(xlist,NTOT,bar,sig,NUSE)
      implicit none
 
      integer NTOT
      real xlist(NTOT)
      real bar
      real sig
      integer NUSE
 
      integer n
      real*8    bsum, ssum
      integer nsum
      integer NIT
 
 
      bar = 0.e0
      sig = 9e9
      do NIT = 1, 20
         bsum = 0.
         ssum = 0.
         nsum = 0.
         do n = 1, NTOT
            if (abs(xlist(n)-bar).le.1.65*sig) then
               bsum = bsum + xlist(n)
               ssum = ssum + abs(xlist(n)-bar)
               nsum = nsum + 1
               endif
            enddo
         if (nsum.gt.0) bar = bsum / nsum
         if (nsum.gt.1) sig = ssum/(nsum-1)*1.1
         enddo
      NUSE = nsum
      if (nsum.le.1) sig = 0.999
 
      return
      end
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/ROUTINES/PHOT/apphot_NAXIS.f"
c****                                         
c*********************************************
                                              
 
c----------------------------------------
c
c this routine will do simple aperture photometry
c about a pixel (i,j)
c
 
      real function apphot_NAXIS(i,j,rap,ri,ro,pix,sky,NAXIS1,NAXIS2)
      implicit none
 
      integer i, j
      real    rap
      integer ri, ro
      integer  NAXIS1,NAXIS2
      real pix(NAXIS1,NAXIS2)
      real    sky
 
 
      integer ii, jj
      real mbar_sky_NAXIS
      real    ftot, ftoth
      integer L, LL, Ls
      real    pl(1000), puse
      real    rl(1000)
      integer il(1000)
      integer jl(1000)
 
      integer Us, Uu
      real    vu(1000), vbar, vsig
      real    abs_rap
 
c------------------------------------------------
c
c special case, saturated flux added up in corral
c
      if (pix(i  ,j  ).gt.0.and.
     .    pix(i-1,j  ).eq.0.and.
     .    pix(i+1,j  ).eq.0.and.
     .    pix(i  ,j+1).eq.0.and.
     .    pix(i  ,j-1).eq.0.and.
     .    pix(i-1,j+1).eq.0.and.
     .    pix(i-1,j-1).eq.0.and.
     .    pix(i+1,j+1).eq.0.and.
     .    pix(i+1,j-1).eq.0) then
          apphot_NAXIS = pix(i,j)
          return
          endif
 
      sky = mbar_sky_NAXIS(i,j,ri,ro,pix,NAXIS1,NAXIS2)
 
      abs_rap = abs(rap)
 
      Ls = 0
      ftot = 0
      do ii = i-int(abs_rap+0.5),i+int(abs_rap+0.5)
      do jj = j-int(abs_rap+0.5),j+int(abs_rap+0.5)
         if (ii.lt. 0001 ) goto 1
         if (jj.lt. 0001 ) goto 1
         if (ii.gt.NAXIS1) goto 1
         if (jj.gt.NAXIS2) goto 1
         if ((ii-i)**2+(jj-j)**2.gt.rap**2) goto 1
         ftot = ftot + pix(ii,jj)-sky
         Ls = Ls + 1
         if (Ls.gt.1000) stop 'Ls.gt.1000 in apphot_NAXIS'
         pl(Ls) = pix(ii,jj)-sky
         rl(Ls) = sqrt(1.*(i-ii)**2+(j-jj)**2)
         il(Ls) = ii
         jl(Ls) = jj
 1       continue
         enddo
         enddo
 
      if (rap.gt.0) then
          apphot_NAXIS = ftot
          return
          endif
 
      ftoth = ftot
      ftot  = 0.
      do L  = 1, Ls
         puse = pl(L)
         if (rl(L).le.2.99) then
c            write(52,218) 1.*il(L), 1.*jl(L), 0.50
             goto 3
             endif
         Us = 0
         do LL = 1, Ls
            if (L.ne.LL) then
               if (abs(rl(L)-rl(LL)).le.0.99) then
                   Us = Us + 1
                   vu(Us) = pl(LL)
                   endif
               endif
            enddo
         call rbarsigs(vu,Us,vbar,vsig,Us,3.0)
         if (abs(pl(L)-vbar).lt.5.0*vsig) then
c            write(52,216) 1.*il(L), 1.*jl(L), 0.35
            else
             puse = vbar
c            write(52,217) 1.*il(L), 1.*jl(L), 0.50
            endif
    3    ftot = ftot + puse
         enddo
 
      apphot_NAXIS = ftot
 
c     write(51,157) ftot, ftoth, i, j
c 157 format(1x,f10.2,1x,f10.2,1x,i4,1x,i4)
      return
      end
 
 
 
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/FITSIO/GEN/query_hdr.f"
c****                                         
c*********************************************
                                              
 
c---------------------------------------------------
c
c This routine checks to see if the header of an image
c contains a particular keyword. If it does, it returns
c its value-string in STREAMX.
c
 
      subroutine query_hdr(filename,FIELDX,streamx)
      implicit none
 
      character*80 filename
 
      character*8  field
      character*20 stream
 
      character*8  fieldx
      character*20 streamx
 
      integer  i
      integer ios, k
 
      character*2880 buff
 
      integer nread
      integer NEXTEND
 
c-----------------------------------------------
 
      !print*,'query_hdr...',FILENAME
      !print*,'    fieldx: ',fieldx
      streamx = ' '
 
      close(10)
      open(10,file=FILENAME,status='old',iostat=ios,
     .  err=900,recl=2880,form='UNFORMATTED',
     .  access='DIRECT')
 
      streamx = 'NULL'
      i = 0
      NREAD   = 0
      NEXTEND = 0
 100  continue
      i = i + 1
      read(10,rec=i,iostat=ios,err=901) buff
      if (ios.lt.0) goto 900
      do k = 0, 35, 1
         field  = buff(k*80+01:k*80+08)
         stream = buff(k*80+11:k*80+31)
         !print*,i,k,field,stream
         if (field.eq.'NEXTEND ') read(stream,*) NEXTEND
         if (field.eq.fieldx) streamx = stream(1:20)
         if (field.eq.'END     ') goto 101
 109     continue
         enddo
      goto 100
 101  continue
      NREAD = NREAD + 1
      if (NREAD.le.1.and.NEXTEND.gt.1) goto 100
      close(10)
      !print*,'    streamx: ',streamx
 
      return
 
  900 continue
      print*,' '
      print*,'query_hdr() FILE OPEN ERROR EXIT. '
      print*,' '
      print*,'ONE OF THE IMAGES WAS NOT IN STANDARD'
      print*,'HST FITS FORMAT.'
      print*,' '
      write(*,'(''PROBLEM FILE: '',a80)') FILENAME
      write(*,'(''      FIELDX: '',a08)') FIELDX
      write(*,'(''      IOSTAT: '',i4 )') IOS
      print*,' '
      stop
 
  901 continue
      print*,' '
      print*,'query_hdr() FILE READ ERROR EXIT. '
      print*,' '
      print*,'ONE OF THE IMAGES WAS NOT IN STANDARD'
      print*,'HST FITS FORMAT.'
      print*,' '
      write(*,'(''PROBLEM FILE: '',a80)') FILENAME
      write(*,'(''      FIELDX: '',a08)') FIELDX
      write(*,'(''      IOSTAT: '',i4 )') IOS
      print*,' '
      stop
 
      end
 
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/FITSIO/GEN/readfits_i2r.f"
c****                                         
c*********************************************
                                              
 
 
      subroutine readfits_i2r(FILE,pix,NX,NY)
      implicit none
 
      character*(*) FILE
      integer NX, NY
      real*4 pix(NX,NY)
 
      character*70 INFO(10)
      common / fitsinfo / INFO
 
      integer naxes
      integer laxis(3)
 
      character*8  field
      character*20 stream
 
      real*4 pixu
      integer nbyte0
      integer nbyteE
      integer nbyte1
      integer nbyte2
      integer nbper
      integer i,ios, k
      integer ii, jj
      integer n
      integer NXU, NYU
 
      character*2880 buffc
      byte buffb(2880)
      equivalence (buffc,buffb)
 
      integer*2 ibuff(1440)
 
      integer ifirst, i1, i2
      integer j
 
      integer np1, np2, npt
      integer nextend
      integer nread
      real bscale, bzero
      integer bitpix
 
      integer r2i
 
      logical DIAG
      data DIAG /.false./
 
 
      character*70 HDR(25)
      common/HDR/HDR
      character*80 FILEU
 
      FILEU = FILE
      do i = 75,2,-1
         if (FILE(i:i+4).eq.'.fits') FILEU = FILE(1:i+4)
         enddo
 
      open(10,file=FILEU,status='old',
     .     err=900,recl=2880,form='UNFORMATTED',
     .     access='DIRECT')
 
      if (DIAG) print*,'...opened'
 
      bscale = 1
      bzero  = 0
 
      naxes = -1
      laxis(1) = 1
      laxis(2) = 1
      laxis(3) = 1
      nextend = 0
 
      do i = 1, 10
         INFO(i) = ' '
         enddo
      do i = 1, 25
         HDR(i) = ' '
         enddo
      HDR(24) = '  1.000 '
 
      i = 0
      nread = 0
 100  continue
      i = i + 1
      read(10,rec=i,iostat=ios) buffc
      if (DIAG) print*,'READREC: ',i
      do k = 0, 35, 1
         if (DIAG) write(*,'(i4,1x,i4,1x,a80)')
     .                   i,k,buffc(k*80+1:k*80+80)
         field  = buffc(k*80+01:k*80+08)
         stream = buffc(k*80+11:k*80+31)
         if (field.eq.'NAXIS   ') read(stream,*) naxes
         if (field.eq.'NAXIS1  ') read(stream,*) laxis(1)
         if (field.eq.'NAXIS2  ') read(stream,*) laxis(2)
         if (field.eq.'NAXIS3  ') read(stream,*) laxis(3)
         if (field.eq.'NEXTEND ') read(stream,*) nextend
         if (field.eq.'BITPIX  ') read(stream,*) bitpix
         if (field.eq.'BSCALE  ') read(stream,*) bscale
         if (field.eq.'BZERO   ') read(stream,*) bzero
 
         if (field.eq.'EXPTIME ') INFO(01) = stream
         if (field.eq.'FILTNAM1') INFO(02) = stream
         if (field.eq.'FILENAME') INFO(03) = stream
         if (field.eq.'DATE-OBS') INFO(04) = stream
         if (field.eq.'TIME-OBS') INFO(05) = stream
         if (field.eq.'DEC_TARG') INFO(06) = stream
         if (field.eq.'RA_TARG ') INFO(07) = stream
         if (field.eq.'PA_V3   ') INFO(08) = stream
         if (field.eq.'PROPOSID') INFO(09) = stream
         if (field.eq.'CCDGAIN ') INFO(10) = stream
 
         if (field.eq.'CRPIX1  ') HDR(01) = stream
         if (field.eq.'CRPIX2  ') HDR(02) = stream
         if (field.eq.'CRVAL1  ') HDR(03) = stream
         if (field.eq.'CRVAL2  ') HDR(04) = stream
         if (field.eq.'CTYPE1  ') HDR(05) = stream
         if (field.eq.'CTYPE2  ') HDR(06) = stream
         if (field.eq.'CD1_1   ') HDR(07) = stream
         if (field.eq.'CD1_2   ') HDR(08) = stream
         if (field.eq.'CD2_1   ') HDR(09) = stream
         if (field.eq.'CD2_2   ') HDR(10) = stream
         if (field.eq.'ORIENTAT') HDR(11) = stream
         if (field.eq.'PA_APER ') HDR(12) = stream
         if (field.eq.'PA_V3   ') HDR(13) = stream
         if (field.eq.'DATE-OBS') HDR(14) = stream
         if (field.eq.'TIME-OBS') HDR(15) = stream
         if (field.eq.'EXPTIME ') HDR(16) = stream
         if (field.eq.'ROOTNAME') HDR(17) = stream
         if (field.eq.'TARGNAME') HDR(18) = stream
         if (field.eq.'RA_TARG ') HDR(19) = stream
         if (field.eq.'DEC_TARG') HDR(20) = stream
         if (field.eq.'PROPOSID') HDR(21) = stream
         if (field.eq.'FILTER1 ') HDR(22) = stream
         if (field.eq.'FILTER2 ') HDR(23) = stream
         if (field.eq.'VAFACTOR') HDR(24) = stream
 
         if (field.eq.'END     ') goto 101
         enddo
      goto 100
 101  continue
 
      nread = nread + 1
      if (DIAG) then
         print*,'----------------------------------------'
         print*,'  NREAD: ',nread
         print*,'NEXTEND: ',nextend
         print*,'  NAXIS: ',naxes
         print*,'  LAXIS: ',laxis(1),laxis(2),laxis(3)
         print*,' BITPIX: ',bitpix
         print*,' BSCALE: ',bscale
         print*,'  BZERO: ',bzero
         endif
 
      ifirst = i+1
      i1 = i
      i2 = i
 
      if (BITPIX.ne.16) then
         print*,'readfits_i2...: '
         print*,'                '
         print*,'   you called a routine to read in an'
         print*,'   unsigned i2 image, and the image you'
         print*,'   gave it has BITPIX = ',BITPIX
         print*,'   '
         print*,'   FILEU: ',FILEU
         print*,'   '
         stop
         endif
 
      nbper = 2*laxis(1)*laxis(2)
      npt   =   laxis(1)*laxis(2)
      nbyte1 = 1
      nbyte2 = nbper
      i1 = i+1 + nbyte1/2880
      i2 = i+1 + nbyte2/2880
 
      if (laxis(1).gt.NX.and.laxis(2).gt.NY) then
         print*,' not enough image space! '
         print*,' '
         print*,'  laxis1: ',laxis(1)
         print*,'      NX: ',NX
         print*,' '
         print*,'  laxis2: ',laxis(2)
         print*,'      NY: ',NY
         print*,' '
         stop
         endif
 
      NXU = laxis(1)
      NYU = laxis(2)
 
      if (DIAG) then
         print*,'     NX: ',NX
         print*,'     NY: ',NY
         print*,'  NBPER: ',nbper
         print*,'  NBYT1: ',nbyte1
         print*,'  NBYT2: ',nbyte2
         print*,' IFIRST: ',ifirst
         print*,'     I1: ',i1
         print*,'     I2: ',i2
         print*,'    NPT: ',NPT
         endif
      do i = i1, i2, 1
         read(10,rec=i,iostat=ios) buffc
         nbyte0 = (i-ifirst)*2880+   1
         nbyteE = (i-ifirst)*2880+2880
         np1 = (nbyte0-nbyte1)/2 + 1
         np2 = (nbyteE-nbyte1)/2 + 1
         np2 = min(np2,npt)
         call buff2pix_i2r(buffb,ibuff,0001,1440)
         do n = np1, np2, 1
            jj = n/NXU   + 1
            ii = n-NXU*(jj-1)
            pixu = ibuff(n-np1+1)*bscale+bzero
            pix(ii,jj) = pixu
            enddo
         enddo
 
      if (DIAG) then
         print*,'  NBPER: ',nbper
         print*,'  NBYT1: ',nbyte1
         print*,'  NBYT2: ',nbyte2
         print*,' IFIRST: ',ifirst
         print*,'     I1: ',i1
         print*,'     I2: ',i2
         print*,'    NPT: ',NPT
         endif
 
 
      return
 
 900  continue
      print*,'READFITS_I2 ERROR'
      print*,'   FILEU: ',FILEU
      stop
 
      end
 
 
 
      subroutine buff2pix_i2r(buff,pix,n1,nt)
      implicit none
      byte buff(2880)
      integer*2 pix(*)
      integer n1,nt
 
      byte b(2)
      integer ii
      equivalence(ii,b)
 
      integer i, npu, nbu
      logical islinux
 
      do i = 1, 1440
         npu = n1+i-1
         nbu = (i-1)*2
         if (.not.(islinux())) then
            b(1) = buff(nbu+1)
            b(2) = buff(nbu+2)
            endif
         if ((islinux())) then
            b(2) = buff(nbu+1)
            b(1) = buff(nbu+2)
            endif
         if (npu.ge.1.and.npu.le.nt) pix(npu) = ii
         enddo
 
      return
      end
 
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/FITSIO/GEN/readfits_i4r.f"
c****                                         
c*********************************************
                                              
 
c--------------------------------------------------------
c
c This routine reads in images stored in integer*4 format...
c
c to use:
c
c   input parameters:
c
c         FILE     is the filename
c         NX,NY    are the dimensions of the image to be
c                  read in
c
c
c  output parameters:
c
c         pix( , ) is an integer*4 image of dimensions NX,NY
c
c
c
      subroutine readfits_i4r(FILE,pix,NX,NY)
      implicit none
 
      integer NX, NY
      character*80 FILE
      real*4 pix(NX,NY)
 
      character*70 INFO(10)
      common / fitsinfo / INFO
 
      integer naxes
      integer laxis(3)
 
      character*8  field
      character*20 stream
 
      integer nbyte0
      integer nbyteE
      integer nbyte1
      integer nbyte2
      integer nbper
      integer i,ios, k
      integer ii, jj
      integer n
 
      character*2880 buffc
      byte buffb(2880)
      equivalence (buffb,buffc)
 
      integer*4 ibuff(720)
 
      integer ifirst, i1, i2
      integer j
 
      integer np1, np2, npt
      integer nextend
      integer nread
      real bscale, bzero
      integer bitpix
 
      logical LINUX
      data LINUX/.true./
      logical DIAG
      data DIAG /.false./
 
 
      character*70 HDR(25)
      common/HDR/HDR
 
      character*80 FILEU
      FILEU = FILE
      do i = 75,2,-1
         if (FILE(i:i+4).eq.'.fits') FILEU = FILE(1:i+4)
         enddo
 
      if (DIAG) then
         print*,'enter readfits...'
         print*,'FILE: ',FILE(1:60)
         endif
 
      open(10,file=FILEU,status='old',
     .     err=900,recl=2880,form='UNFORMATTED',
     .     access='DIRECT')
 
      if (DIAG) print*,'...opened'
 
      bscale = 1
      bzero  = 0
 
      naxes = -1
      laxis(1) = 1
      laxis(2) = 1
      laxis(3) = 1
      nextend = 0
 
      do i = 1, 10
         INFO(i) = ' '
         enddo
      do i = 1, 23
         HDR(i) = ' '
         enddo
 
      i = 0
      nread = 0
 100  continue
      i = i + 1
      read(10,rec=i,iostat=ios) buffc
      if (DIAG) print*,'READREC: ',i
      do k = 0, 35, 1
         if (DIAG) write(*,'(i4,1x,i4,1x,a80)')
     .                   i,k,buffc(k*80+1:k*80+80)
         field  = buffc(k*80+01:k*80+08)
         stream = buffc(k*80+11:k*80+31)
         if (field.eq.'NAXIS   ') read(stream,*) naxes
         if (field.eq.'NAXIS1  ') read(stream,*) laxis(1)
         if (field.eq.'NAXIS2  ') read(stream,*) laxis(2)
         if (field.eq.'NAXIS3  ') read(stream,*) laxis(3)
         if (field.eq.'NEXTEND ') read(stream,*) nextend
         if (field.eq.'BITPIX  ') read(stream,*) bitpix
         if (field.eq.'BSCALE  ') read(stream,*) bscale
         if (field.eq.'BZERO   ') read(stream,*) bzero
 
         if (field.eq.'EXPTIME ') INFO(01) = stream
         if (field.eq.'FILTNAM1'.and.stream(1:1).eq.'F')
     .                            INFO(02) = stream
         if (field.eq.'FILTNAM2'.and.stream(1:1).eq.'F')
     .                            INFO(02) = stream
         if (field.eq.'FILTER1 '.and.stream(1:1).eq.'F')
     .                            INFO(02) = stream
         if (field.eq.'FILTER2 '.and.stream(1:1).eq.'F')
     .                            INFO(02) = stream
         if (field.eq.'FILENAME') INFO(03) = stream
         if (field.eq.'DATE-OBS') INFO(04) = stream
         if (field.eq.'TIME-OBS') INFO(05) = stream
         if (field.eq.'DEC_TARG') INFO(06) = stream
         if (field.eq.'RA_TARG ') INFO(07) = stream
         if (field.eq.'PA_V3   ') INFO(08) = stream
         if (field.eq.'PROPOSID') INFO(09) = stream
         if (field.eq.'CCDGAIN ') INFO(10) = stream
 
         if (field.eq.'CRPIX1  ') HDR(01) = stream
         if (field.eq.'CRPIX2  ') HDR(02) = stream
         if (field.eq.'CRVAL1  ') HDR(03) = stream
         if (field.eq.'CRVAL2  ') HDR(04) = stream
         if (field.eq.'CTYPE1  ') HDR(05) = stream
         if (field.eq.'CTYPE2  ') HDR(06) = stream
         if (field.eq.'CD1_1   ') HDR(07) = stream
         if (field.eq.'CD1_2   ') HDR(08) = stream
         if (field.eq.'CD2_1   ') HDR(09) = stream
         if (field.eq.'CD2_2   ') HDR(10) = stream
         if (field.eq.'ORIENTAT') HDR(11) = stream
         if (field.eq.'PA_APER ') HDR(12) = stream
         if (field.eq.'PA_V3   ') HDR(13) = stream
         if (field.eq.'DATE-OBS') HDR(14) = stream
         if (field.eq.'TIME-OBS') HDR(15) = stream
         if (field.eq.'EXPTIME ') HDR(16) = stream
         if (field.eq.'ROOTNAME') HDR(17) = stream
         if (field.eq.'TARGNAME') HDR(18) = stream
         if (field.eq.'RA_TARG ') HDR(19) = stream
         if (field.eq.'DEC_TARG') HDR(20) = stream
         if (field.eq.'PROPOSID') HDR(21) = stream
         if (field.eq.'FILTER1 ') HDR(22) = stream
         if (field.eq.'FILTER2 ') HDR(23) = stream
         if (field.eq.'VAFACTOR') HDR(24) = stream
 
 
         if (field.eq.'END     ') goto 101
         enddo
      goto 100
 101  continue
 
 
      if (laxis(1).ne.NX.or.laxis(2).ne.NY) then
         print*,'FITS image not the expected dimensions: '
         print*,'    input NX: ',nx
         print*,'    input NY: ',ny
         print*,'              '
         print*,'    laxis(1): ',laxis(1)
         print*,'    laxis(2): ',laxis(2)
         print*,'  '
         stop
         endif
 
      nread = nread + 1
      if (DIAG) then
         print*,'----------------------------------------'
         print*,'  NREAD: ',nread
         print*,'NEXTEND: ',nextend
         print*,'  NAXIS: ',naxes
         print*,'  LAXIS: ',laxis(1),laxis(2),laxis(3)
         print*,' BITPIX: ',bitpix
         print*,' BSCALE: ',bscale
         print*,'  BZERO: ',bzero
         endif
 
      ifirst = i+1
      i1 = i
      i2 = i
 
      if (BITPIX.ne.32) then
         print*,'                '
         print*,'readfits_i4...: '
         print*,'                '
         print*,'   you called a routine to read in a'
         print*,'   long i*4 image, and the image you'
         print*,'   gave it has BITPIX = ',BITPIX
         print*,'   '
         print*,'   FILE: ',FILE
         print*,'   '
         stop
         endif
 
      nbper = 4*laxis(1)*laxis(2)
      npt   =   laxis(1)*laxis(2)
      nbyte1 = 1
      nbyte2 = nbper
      i1 = i+1 + nbyte1/2880
      i2 = i+1 + nbyte2/2880
 
      do i = i1, i2, 1
         read(10,rec=i,iostat=ios) buffc
         nbyte0 = (i-ifirst)*2880+   1
         nbyteE = (i-ifirst)*2880+2880
         np1 = (nbyte0-nbyte1)/4 + 1
         np2 = (nbyteE-nbyte1)/4 + 1
         np2 = min(np2,npt)
         call buff2pix_i4r(buffb,ibuff,0001,0720)
         do n = np1, np2, 1
            jj = n/NX   + 1
            ii = n-NX*(jj-1)
            pix(ii,jj) = ibuff(n-np1+1)*bscale+bzero
            enddo
         enddo
 
      if (DIAG) then
         print*,'  NBPER: ',nbper
         print*,'  NBYT1: ',nbyte1
         print*,'  NBYT2: ',nbyte2
         print*,' IFIRST: ',ifirst
         print*,'     I1: ',i1
         print*,'     I2: ',i2
         print*,'    NPT: ',NPT
         endif
 
      return
 
 900  continue
      print*,'READFITS_I4R ERROR'
      print*,' FILEU: ',FILEU
      stop
 
      end
 
 
      subroutine buff2pix_i4r(buff,pix,n1,nt)
      implicit none
      byte buff(2880)
      integer*4 pix(*)
      integer n1,nt
 
      byte b(4)
      integer ii
      equivalence(ii,b)
 
      integer i, npu, nbu
 
      do i = 1, 720
         npu = n1+i-1
         nbu = (i-1)*4
         if (.not.(_LINUX_)) then
            b(1) = buff(nbu+1)
            b(2) = buff(nbu+2)
            b(3) = buff(nbu+3)
            b(4) = buff(nbu+4)
            endif
         if ((_LINUX_)) then
            b(4) = buff(nbu+1)
            b(3) = buff(nbu+2)
            b(2) = buff(nbu+3)
            b(1) = buff(nbu+4)
            endif
         if (npu.ge.1.and.npu.le.nt) pix(npu) = ii
         enddo
 
      return
      end
 
 
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/FITSIO/GEN/readfits_r4.f"
c****                                         
c*********************************************
                                              
 
c--------------------------------------------------
c
c this just reads in an r4 fits image
c
 
      subroutine readfits_r4(FILE,pix,NDIMX,NDIMY)
      implicit none
 
      character*(*) FILE
      integer NDIMX,NDIMY
      real    pix(NDIMX,NDIMY)
 
      character*199 FILEU
      character*070 INFO(10)
 
      integer naxes
      integer laxis(3)
 
      character*8  field
      character*70 stream
 
      integer nbyte0
      integer nbyteE
      integer nbyte1
      integer nbyte2
      integer nbper
      integer i,ios, k
      integer j
 
      character*2880 buffc
      byte   buffb(2880)
      equivalence (buffc,buffb)
 
      real*4 buffr(0720)
      integer ii,nn,nx,ny
      integer nxx, nyy
      integer ifirst, i1, i2
 
      integer np1, np2, npt
      integer nextend
      integer nread
      real bscale, bzero
      integer bitpix
 
      logical DIAG
      data DIAG /.false./
 
      character*70 HDR(25)
c     common/HDR/HDR
 
      logical extend_tf
 
      FILEU = FILE
      do i = 195,2,-1
         if (FILE(i:i+4).eq.'.fits') FILEU = FILE(1:i+4)
         enddo
 
      do i = 1, 25
         HDR(i) = ' '
         enddo
 
 
      if (DIAG) then
         print*,'enter readfits_r4...'
         print*,'FILE: ',FILE(1:60)
         endif
 
      open(10,file=FILEU,status='old',
     .     err=900,recl=2880,form='UNFORMATTED',
     .     access='DIRECT')
 
      if (DIAG) print*,'...opened'
 
      naxes = -1
      laxis(1) = 1
      laxis(2) = 1
      laxis(3) = 1
      nextend = 0
 
      do i = 1, 10
         INFO(i) = ' '
         enddo
 
      extend_tf = .false.
 
      i = 0
      nread = 0
 100  continue
      i = i + 1
      read(10,rec=i,iostat=ios) buffc
      if (DIAG) print*,'READREC: ',i
      do k = 0, 35, 1
         if (DIAG) write(*,'(i4,1x,i4,1x,a80)')
     .                   i,k,buffc(k*80+1:k*80+80)
         do j = 1, 80
            if (ichar(buffc(k*80+j:k*80+j)).eq.0)
     .                buffc(k*80+j:k*80+j) = ' '
            enddo
         field  = buffc(k*80+01:k*80+08)
         stream = buffc(k*80+10:k*80+79)
         if (DIAG) write(*,'(''   FIELD : '',8a)') field
         if (DIAG) write(*,'(''   STREAM: '',20a)') stream
         if (DIAG) write(*,'(''   STREAM: '',70a)') stream
         if (field.eq.'EXTEND  ') read(stream,*) extend_tf
         if (field.eq.'NAXIS   ') read(stream,*) naxes
         if (field.eq.'NAXIS1  ') read(stream,*) laxis(1)
         if (field.eq.'NAXIS2  ') read(stream,*) laxis(2)
         if (field.eq.'NAXIS3  ') read(stream,*) laxis(3)
         if (field.eq.'NEXTEND ') read(stream,*) nextend
         if (field.eq.'BITPIX  ') read(stream,*) bitpix
         if (field.eq.'BSCALE  ') read(stream,*) bscale
         if (field.eq.'BZERO   ') read(stream,*) bzero
 
         if (field.eq.'EXPTIME ') INFO(1) = stream
         if (field.eq.'FILTNAM1') INFO(2) = stream
         if (field.eq.'FILENAME') INFO(3) = stream
         if (field.eq.'DATE-OBS') INFO(4) = stream
         if (field.eq.'TIME-OBS') INFO(5) = stream
         if (field.eq.'DEC_TARG') INFO(6) = stream
         if (field.eq.'RA_TARG ') INFO(7) = stream
         if (field.eq.'PA_V3   ') INFO(8) = stream
         if (field.eq.'PROPOSID') INFO(9) = stream
 
         if (field.eq.'CRPIX1  ') HDR(01) = stream
         if (field.eq.'CRPIX2  ') HDR(02) = stream
         if (field.eq.'CRVAL1  ') HDR(03) = stream
         if (field.eq.'CRVAL2  ') HDR(04) = stream
         if (field.eq.'CTYPE1  ') HDR(05) = stream
         if (field.eq.'CTYPE2  ') HDR(06) = stream
         if (field.eq.'CD1_1   ') HDR(07) = stream
         if (field.eq.'CD1_2   ') HDR(08) = stream
         if (field.eq.'CD2_1   ') HDR(09) = stream
         if (field.eq.'CD2_2   ') HDR(10) = stream
         if (field.eq.'ORIENTAT') HDR(11) = stream
         if (field.eq.'PA_APER ') HDR(12) = stream
         if (field.eq.'PA_V3   ') HDR(13) = stream
         if (field.eq.'DATE-OBS') HDR(14) = stream
         if (field.eq.'TIME-OBS') HDR(15) = stream
         if (field.eq.'EXPTIME ') HDR(16) = stream
         if (field.eq.'ROOTNAME') HDR(17) = stream
         if (field.eq.'TARGNAME') HDR(18) = stream
         if (field.eq.'RA_TARG ') HDR(19) = stream
         if (field.eq.'DEC_TARG') HDR(20) = stream
         if (field.eq.'PROPOSID') HDR(21) = stream
         if (field.eq.'FILTER1 ') HDR(22) = stream
         if (field.eq.'FILTER2 ') HDR(23) = stream
         if (field.eq.'VAFACTOR') HDR(24) = stream
 
         if (field.eq.'END     ') goto 101
         enddo
      goto 100
 101  continue
 
      nread = nread + 1
 
      if (extend_tf.and.NREAD.eq.1.and.NAXES.eq.0) goto 100 ! read the first extension...
 
      if (DIAG) then
         print*,'----------------------------------------'
         print*,'  NREAD: ',nread
         print*,'NEXTEND: ',nextend
         print*,'  NAXIS: ',naxes
         print*,'  LAXIS: ',laxis(1),laxis(2),laxis(3)
         print*,' BITPIX: ',bitpix
         print*,' BSCALE: ',bscale
         print*,'  BZERO: ',bzero
         endif
 
      ifirst = i+1
      i1 = i
      i2 = i
 
      nbper  = 4*laxis(1)*laxis(2)
      npt    = laxis(1)*laxis(2)
      nbyte1 = 1
      nbyte2 = nbper
      i1 = i+1 + nbyte1/2880
      i2 = i+1 + nbyte2/2880
 
      if (BITPIX.ne.-32) then
         print*,'   FILEU: ',FILEU(1:40)
         print*,'   FILE : ',FILE(1:40)
         print*,'BITPIX.ne.-32... unreal!'
         stop
         endif
 
      do i = 01, NDIMX
      do j = 01, NDIMY
         pix(i,j) = 0.
         enddo
         enddo
 
      do i = i1, i2, 1
         read(10,rec=i,iostat=ios) buffc
         nbyte0 = (i-ifirst)*2880+   1
         nbyteE = (i-ifirst)*2880+2880
         np1 = (nbyte0-nbyte1)/4 + 1
         np2 = (nbyteE-nbyte1)/4 + 1
         call buff2pix_r4(buffb,buffr,0001,0720)
         do ii = 001, 720
            nn = np1 + (ii-1)
            ny = 1 + (nn-1)/laxis(1)
            nx = nn-(ny-1)*laxis(1)
            if (nx.ge.001.and.nx.le.NDIMX.and.
     .          ny.ge.001.and.ny.le.NDIMY) then
                pix(nx,ny) = buffr(ii)
                nxx = nx
                nyy = ny
                endif
            enddo
         if (ny.gt.NDIMY) goto 899
 
c        if (DIAG) write(*,1115) i,np1,np2,npt,nxx,nyy
 1115    format(1x,i8,1x,i10,1x,i10,1x,i10,1x,2i6)
         enddo
 
  899 close(10)
c     if (DIAG) write(*,1115) i,np1,np2,npt,nxx,nyy
 
 
 
      return
 
  900 continue
      print*,'                '
      print*,'READFITS ERROR: '
      print*,'                '
      write(*,'(''   could not read in file:   '',80a)') FILEU
      print*,'                '
      stop
 
      end
 
 
 
      subroutine buff2pix_r4(buff,pix,n1,nt)
      implicit none
      byte buff(2880)
      real pix(*)
      integer n1,nt
 
      byte b(4)
      real r
      equivalence(r,b)
 
      integer i, npu, nbu
      logical islinux
 
      do i = 1, 720
         npu = n1+i-1
         nbu = (i-1)*4
         if (.not.islinux()) then
            b(1) = buff(nbu+1)
            b(2) = buff(nbu+2)
            b(3) = buff(nbu+3)
            b(4) = buff(nbu+4)
            endif
         if (islinux()) then
            b(4) = buff(nbu+1)
            b(3) = buff(nbu+2)
            b(2) = buff(nbu+3)
            b(1) = buff(nbu+4)
            endif
         if (npu.ge.1.and.npu.le.nt) pix(npu) = r
         enddo
 
      return
      end
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/FITSIO/GEN/readfits_r4e.f"
c****                                         
c*********************************************
                                              
 
c-----------------------------------------------------------------
c
c reads an r4 fits image, with extensions (reads in one extension)
c
 
      subroutine readfits_r4e(FILE,pix,NDIMX,NDIMY,NEXTENU)
      implicit none
 
      character*(*) FILE
      integer NDIMX,NDIMY
      real    pix(NDIMX,NDIMY)
      integer NEXTENU
 
      character*199 FILEU
      character*70 INFO(10)
      common / fitsinfo / INFO
 
      integer naxes
      integer laxis(3)
      common/laxis3_/laxis
 
      integer NXF
      integer NYF
 
      character*8  field
      character*40 stream
 
      integer nbyte0
      integer nbyteE
      integer nbyte1
      integer nbyte2
      integer nbper
      integer i,ios, k
      integer j
 
      character*2880 buffc
      byte buffb(2880)
      equivalence (buffc,buffb)
 
      integer ifirst, i1, i2
 
      integer np1, np2, npt
      integer nextend
      integer nread
      real bscale, bzero
      integer bitpix
 
      character*70 HDR(25)
      common/HDR/HDR
 
      logical DIAG
      data DIAG /.false./
      integer NEND
      integer iend
 
      if (DIAG) then
         print*,'enter readfits_r4e...... '
         print*,'FILEi: ',FILE(1:60)
         endif
 
      FILEU = FILE
      iend = 0
      do i = 196,1,-1
         if (DIAG) print*,i,iend,FILE(i:i+4)
         if (FILE(i:i+4).eq.'.fits') iend = i+4
         enddo
      if (DIAG) then
         print*,'iend: ',iend
         endif
      if (iend.eq.0) stop 'NO .fits in FILENAME'
      FILEU = FILE(1:iend)
 
      if (DIAG) then
         print*,'purge...'
         print*,'FILEu: ',FILEU
         endif
 
      open(10,file=FILEU,status='old',
     .     err=900,recl=2880,form='UNFORMATTED',
     .     access='DIRECT')
 
      if (DIAG) print*,'...opened ',FILEU
 
      naxes = -1
      laxis(1) = 1
      laxis(2) = 1
      laxis(3) = 1
      nextend = 0
 
      do i = 1, 10
         INFO(i) = ' '
         enddo
 
      BSCALE = 1.0
      BZERO  = 0.0
 
      NEND = 0
      i = 0
      nread = 0
 100  continue
      i = i + 1
      if (DIAG) print*,'READREC: ',i
      read(10,rec=i,iostat=ios) buffc
      do k = 0, 35, 1
         if (DIAG) write(*,'(i4,1x,i4,1x,a80)')
     .                   i,k,buffc(k*80+1:k*80+80)
         field  = buffc(k*80+01:k*80+08)
         stream = buffc(k*80+11:k*80+51)
         if (field.eq.'NAXIS   ') read(stream,*) naxes
         if (field.eq.'NAXIS1  ') read(stream,*) laxis(1)
         if (field.eq.'NAXIS2  ') read(stream,*) laxis(2)
         if (field.eq.'NAXIS3  ') read(stream,*) laxis(3)
         if (field.eq.'NEXTEND ') read(stream,*) nextend
         if (field.eq.'BITPIX  ') read(stream,*) bitpix
         if (field.eq.'BSCALE  ') read(stream,*) bscale
         if (field.eq.'BZERO   ') read(stream,*) bzero
 
         if (field.eq.'EXPTIME ') INFO(1) = stream
         if (field.eq.'FILTER  ') INFO(2) = stream
         if (field.eq.'FILTNAM1') INFO(2) = stream
         if (field.eq.'FILENAME') INFO(3) = stream
         if (field.eq.'DATE-OBS') INFO(4) = stream
         if (field.eq.'TIME-OBS') INFO(5) = stream
         if (field.eq.'DEC_TARG') INFO(6) = stream
         if (field.eq.'RA_TARG ') INFO(7) = stream
         if (field.eq.'PA_V3   ') INFO(8) = stream
         if (field.eq.'PROPOSID') INFO(9) = stream
 
         if (field.eq.'CRPIX1  ') HDR(01) = stream
         if (field.eq.'CRPIX2  ') HDR(02) = stream
         if (field.eq.'CRVAL1  ') HDR(03) = stream
         if (field.eq.'CRVAL2  ') HDR(04) = stream
         if (field.eq.'CTYPE1  ') HDR(05) = stream
         if (field.eq.'CTYPE2  ') HDR(06) = stream
         if (field.eq.'CD1_1   ') HDR(07) = stream
         if (field.eq.'CD1_2   ') HDR(08) = stream
         if (field.eq.'CD2_1   ') HDR(09) = stream
         if (field.eq.'CD2_2   ') HDR(10) = stream
         if (field.eq.'ORIENTAT') HDR(11) = stream
         if (field.eq.'PA_APER ') HDR(12) = stream
         if (field.eq.'PA_V3   ') HDR(13) = stream
         if (field.eq.'DATE-OBS') HDR(14) = stream
         if (field.eq.'TIME-OBS') HDR(15) = stream
         if (field.eq.'EXPTIME ') HDR(16) = stream
         if (field.eq.'ROOTNAME') HDR(17) = stream
         if (field.eq.'TARGNAME') HDR(18) = stream
         if (field.eq.'RA_TARG ') HDR(19) = stream
         if (field.eq.'DEC_TARG') HDR(20) = stream
         if (field.eq.'PROPOSID') HDR(21) = stream
         if (field.eq.'FILTER ')  HDR(22) = stream
         if (field.eq.'FILTER1 ') HDR(22) = stream
         if (field.eq.'FILTER2 ') HDR(23) = stream
         if (field.eq.'VAFACTOR') HDR(24) = stream
 
         if (field.eq.'END     ') then
           !print*,' ---> NEND: ',NEND,NEXTENU
            if (NEND.eq.NEXTENU) goto 101
            NEND = NEND + 1
            endif
         enddo
      goto 100
 101  continue
 
      nread = nread + 1
      if (DIAG) then
         print*,' '
         print*,'----------------------------------------'
         print*,'  NREAD: ',nread
         print*,'NEXTEND: ',nextend
         print*,'  NAXIS: ',naxes
         print*,'  LAXIS: ',laxis(1),laxis(2),laxis(3)
         print*,' BITPIX: ',bitpix
         print*,' BSCALE: ',bscale
         print*,'  BZERO: ',bzero
         print*,'  NDIMX: ',NDIMX
         print*,'  NDIMY: ',NDIMY
         print*,' '
         endif
 
      ifirst = i+1
      i1 = i
      i2 = i
 
      NXF = laxis(1)
      NYF = laxis(2)
 
      nbper  = 4*laxis(1)*laxis(2)
      npt    = laxis(1)*laxis(2)
      nbyte1 = 1
      nbyte2 = nbper
      i1 = i+1 + nbyte1/2880
      i2 = i+1 + nbyte2/2880
 
      if (BITPIX.eq.-64) then
         print*,'from readfits_r4e... '
         print*,'     BITPIX: ',BITPIX
         print*,'call readfits_r8e... '
         print*,'                   '
         call readfits_r8e_r4(FILE,pix,NDIMX,NDIMY,NEXTENU)
         return
         endif
 
      if (BITPIX.ne.-32) then
         print*,'readfits_r4e...'
         print*,'BITPIX: ',BITPIX
         print*,'prob'
         stop
         endif
 
      do i = i1, i2, 1
         read(10,rec=i,iostat=ios) buffc
         nbyte0 = (i-ifirst)*2880+   1
         nbyteE = (i-ifirst)*2880+2880
         np1 = (nbyte0-nbyte1)/4 + 1
         np2 = (nbyteE-nbyte1)/4 + 1
c        call buff2pix_r4(buffb,pix,np1,npt)
         call buff2pix_r4_edge(buffb,pix,np1,npt,
     .                         NDIMX,NDIMY,laxis(1),laxis(2))
         if (DIAG) write(*,1115) i,np1,np2,npt
 1115    format(1x,i8,1x,i10,1x,i10,1x,i10)
         enddo
 
      close(10)
      if (DIAG) print*,'...closed ',FILEU
      return
 
 900  continue
      print*,'READFITS ERROR'
      stop
 
      end
 
 
c------------------------------------------------------
c
c
      subroutine buff2pix_r4_edge(buff,pix,n1,nt,
     .                            NXP,NYP,NXF,NYF)
      implicit none
c     character buff(2880)
      byte buff(2880)
      integer NXP,NYP
      real pix(NXP,NYP)
      integer n1,nt
      integer NXF,NYF
 
      real pbuff(720)
      common /sneaky/pbuff
      integer i
      integer npu, nbu
      integer NX, NY
 
      byte b(4)
      real r
      equivalence(r,b)
      logical islinux
 
      do i = 1, 720
         npu = n1+i-1
         nbu = (i-1)*4
         if (npu.ge.1.and.npu.le.nt) then
            NX = npu - (npu-1)/NXF*NXF
            NY = 1 + (npu-1)/NXF
            if (NX.le.NXP.and.NY.le.NYP) then
               if (.not.islinux()) then
                  b(1) = buff(nbu+1)
                  b(2) = buff(nbu+2)
                  b(3) = buff(nbu+3)
                  b(4) = buff(nbu+4)
                  endif
               if (islinux()) then
                  b(4) = buff(nbu+1)
                  b(3) = buff(nbu+2)
                  b(2) = buff(nbu+3)
                  b(1) = buff(nbu+4)
                  endif
               if (npu.ge.1.and.npu.le.nt) pix(NX,NY) = r
               endif
            endif
         enddo
 
      return
      end
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/FITSIO/GEN/readfits_r8e_r4.f"
c****                                         
c*********************************************
                                              
 
c-----------------------------------------------------------------
c
c reads an r4 fits image, with extensions (reads in one extension)
c
 
      subroutine readfits_r8e_r4(FILE,pix,NDIMX,NDIMY,NEXTENU)
      implicit none
 
      character*(*) FILE
      integer NDIMX,NDIMY
      real*4  pix(NDIMX,NDIMY)
      integer NEXTENU
 
      character*199 FILEU
      character*70 INFO(10)
      common / fitsinfo / INFO
 
      integer naxes
      integer laxis(3)
      common/laxis3_/laxis
 
      integer NXF
      integer NYF
 
      character*8  field
      character*40 stream
 
      integer nbyte0
      integer nbyteE
      integer nbyte1
      integer nbyte2
      integer nbper
      integer i,ios, k
      integer j
 
      character*2880 buffc
      byte buffb(2880)
      equivalence (buffc,buffb)
 
      integer ifirst, i1, i2
 
      integer np1, np2, npt
      integer nextend
      integer nread
      real bscale, bzero
      integer bitpix
 
      character*70 HDR(25)
      common/HDR/HDR
 
      logical DIAG
      data DIAG /.false./
      integer NEND
      integer iend
 
      if (DIAG) then
         print*,'enter readfits_r4e...... '
         print*,'FILEi: ',FILE(1:60)
         endif
 
      FILEU = FILE
      iend = 0
      do i = 196,1,-1
         if (DIAG) print*,i,iend,FILE(i:i+4)
         if (FILE(i:i+4).eq.'.fits') iend = i+4
         enddo
      if (DIAG) then
         print*,'iend: ',iend
         endif
      if (iend.eq.0) stop 'NO .fits in FILENAME'
      FILEU = FILE(1:iend)
 
      if (DIAG) then
         print*,'purge...'
         print*,'FILEu: ',FILEU
         endif
 
      open(10,file=FILEU,status='old',
     .     err=900,recl=2880,form='UNFORMATTED',
     .     access='DIRECT')
 
      if (DIAG) print*,'...opened ',FILEU
 
      naxes = -1
      laxis(1) = 1
      laxis(2) = 1
      laxis(3) = 1
      nextend = 0
 
      do i = 1, 10
         INFO(i) = ' '
         enddo
 
      BSCALE = 1.0
      BZERO  = 0.0
 
      NEND = 0
      i = 0
      nread = 0
 100  continue
      i = i + 1
      if (DIAG) print*,'READREC: ',i
      read(10,rec=i,iostat=ios) buffc
      do k = 0, 35, 1
         if (DIAG) write(*,'(i4,1x,i4,1x,a80)')
     .                   i,k,buffc(k*80+1:k*80+80)
         field  = buffc(k*80+01:k*80+08)
         stream = buffc(k*80+11:k*80+51)
         if (field.eq.'NAXIS   ') read(stream,*) naxes
         if (field.eq.'NAXIS1  ') read(stream,*) laxis(1)
         if (field.eq.'NAXIS2  ') read(stream,*) laxis(2)
         if (field.eq.'NAXIS3  ') read(stream,*) laxis(3)
         if (field.eq.'NEXTEND ') read(stream,*) nextend
         if (field.eq.'BITPIX  ') read(stream,*) bitpix
         if (field.eq.'BSCALE  ') read(stream,*) bscale
         if (field.eq.'BZERO   ') read(stream,*) bzero
 
         if (field.eq.'EXPTIME ') INFO(1) = stream
         if (field.eq.'FILTER  ') INFO(2) = stream
         if (field.eq.'FILTNAM1') INFO(2) = stream
         if (field.eq.'FILENAME') INFO(3) = stream
         if (field.eq.'DATE-OBS') INFO(4) = stream
         if (field.eq.'TIME-OBS') INFO(5) = stream
         if (field.eq.'DEC_TARG') INFO(6) = stream
         if (field.eq.'RA_TARG ') INFO(7) = stream
         if (field.eq.'PA_V3   ') INFO(8) = stream
         if (field.eq.'PROPOSID') INFO(9) = stream
 
         if (field.eq.'CRPIX1  ') HDR(01) = stream
         if (field.eq.'CRPIX2  ') HDR(02) = stream
         if (field.eq.'CRVAL1  ') HDR(03) = stream
         if (field.eq.'CRVAL2  ') HDR(04) = stream
         if (field.eq.'CTYPE1  ') HDR(05) = stream
         if (field.eq.'CTYPE2  ') HDR(06) = stream
         if (field.eq.'CD1_1   ') HDR(07) = stream
         if (field.eq.'CD1_2   ') HDR(08) = stream
         if (field.eq.'CD2_1   ') HDR(09) = stream
         if (field.eq.'CD2_2   ') HDR(10) = stream
         if (field.eq.'ORIENTAT') HDR(11) = stream
         if (field.eq.'PA_APER ') HDR(12) = stream
         if (field.eq.'PA_V3   ') HDR(13) = stream
         if (field.eq.'DATE-OBS') HDR(14) = stream
         if (field.eq.'TIME-OBS') HDR(15) = stream
         if (field.eq.'EXPTIME ') HDR(16) = stream
         if (field.eq.'ROOTNAME') HDR(17) = stream
         if (field.eq.'TARGNAME') HDR(18) = stream
         if (field.eq.'RA_TARG ') HDR(19) = stream
         if (field.eq.'DEC_TARG') HDR(20) = stream
         if (field.eq.'PROPOSID') HDR(21) = stream
         if (field.eq.'FILTER ')  HDR(22) = stream
         if (field.eq.'FILTER1 ') HDR(22) = stream
         if (field.eq.'FILTER2 ') HDR(23) = stream
         if (field.eq.'VAFACTOR') HDR(24) = stream
 
         if (field.eq.'END     ') then
           !print*,' ---> NEND: ',NEND,NEXTENU
            if (NEND.eq.NEXTENU) goto 101
            NEND = NEND + 1
            endif
         enddo
      goto 100
 101  continue
 
      nread = nread + 1
      if (DIAG) then
         print*,' '
         print*,'----------------------------------------'
         print*,'  NREAD: ',nread
         print*,'NEXTEND: ',nextend
         print*,'  NAXIS: ',naxes
         print*,'  LAXIS: ',laxis(1),laxis(2),laxis(3)
         print*,' BITPIX: ',bitpix
         print*,' BSCALE: ',bscale
         print*,'  BZERO: ',bzero
         print*,'  NDIMX: ',NDIMX
         print*,'  NDIMY: ',NDIMY
         print*,' '
         endif
 
      ifirst = i+1
      i1 = i
      i2 = i
 
      NXF = laxis(1)
      NYF = laxis(2)
 
      nbper  = 8*laxis(1)*laxis(2)
      npt    = laxis(1)*laxis(2)
      nbyte1 = 1
      nbyte2 = nbper
      i1 = i+1 + nbyte1/2880
      i2 = i+1 + nbyte2/2880
 
      if (BITPIX.ne.-64) then
         print*,'readfits_r8e...'
         print*,'BITPIX: ',BITPIX
         print*,'prob'
         stop
         endif
 
      if (DIAG) then
         print*,'          '
         print*,'do i ---  ',i1,i2
         print*,'   nbper: ',nbper
         print*,'     npt: ',npt
         print*,'          '
         endif
 
      do i = i1, i2, 1
         read(10,rec=i,iostat=ios) buffc
         nbyte0 = (i-ifirst)*2880+   1
         nbyteE = (i-ifirst)*2880+2880
         np1 = (nbyte0-nbyte1)/8 + 1
         np2 = (nbyteE-nbyte1)/8 + 1
         call buff2pix_r8_edge(buffb,pix,np1,npt,
     .                         NDIMX,NDIMY,laxis(1),laxis(2))
         if (DIAG) write(*,1115) i,np1,np2,npt
 1115    format(1x,i8,1x,i10,1x,i10,1x,i10)
         enddo
 
      if (DIAG) then
          print*,' '
          do j = 10, 01, -1
             write(*,'(10x,i2,1x,10f8.2)') j, (pix(i,j),i=1,10)
             enddo
          print*,' '
          endif
 
      close(10)
      if (DIAG) print*,'...closed ',FILEU
      return
 
 900  continue
      print*,'READFITS ERROR'
      stop
 
      end
 
 
c------------------------------------------------------
c
c
      subroutine buff2pix_r8_edge(buff,pix,n1,nt,
     .                            NXP,NYP,NXF,NYF)
      implicit none
      byte buff(2880)
      integer NXP,NYP
      real*4  pix(NXP,NYP)
      integer n1,nt
      integer NXF,NYF
 
      real pbuff(360)
      integer i
      integer npu, nbu
      integer NX, NY
 
      byte b(8)
      real*8 r8
      equivalence(r8,b)
      logical islinux
 
      do i = 1, 360
         npu = n1+i-1
         nbu = (i-1)*8
         if (npu.ge.1.and.npu.le.nt) then
            NX = npu - (npu-1)/NXF*NXF
            NY = 1 + (npu-1)/NXF
            if (NX.le.NXP.and.NY.le.NYP) then
               if (.not.islinux()) then
                  b(1) = buff(nbu+1)
                  b(2) = buff(nbu+2)
                  b(3) = buff(nbu+3)
                  b(4) = buff(nbu+4)
                  b(5) = buff(nbu+5)
                  b(6) = buff(nbu+6)
                  b(7) = buff(nbu+7)
                  b(8) = buff(nbu+8)
                  endif
               if (islinux()) then
                  b(8) = buff(nbu+1)
                  b(7) = buff(nbu+2)
                  b(6) = buff(nbu+3)
                  b(5) = buff(nbu+4)
                  b(4) = buff(nbu+5)
                  b(3) = buff(nbu+6)
                  b(2) = buff(nbu+7)
                  b(1) = buff(nbu+8)
                  endif
               if (npu.ge.1.and.npu.le.nt) pix(NX,NY) = r8
               endif
            endif
         enddo
 
      return
      end
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/FITSIO/GEN/writfits_r4.f"
c****                                         
c*********************************************
                                              
 
c-----------------------------------------------------
c
c this just writes a real*4 fits image
c
 
      subroutine writfits_r4(FILE,pix,PXDIMX,PXDIMY)
      implicit none
 
      character*(*) FILE
      integer PXDIMX,PXDIMY
      real    pix(PXDIMX,PXDIMY)
 
      integer nbyte0
      integer nbyteE
      integer nbyte1
      integer nbyte2
      integer nbper
      integer i,ios
 
      character*2880 buffc
      byte buffb(2880)
      equivalence (buffb,buffc)
 
      integer ifirst, i1, i2
 
      integer np1, np2, npt
      integer k
 
      character*80 FILEU
      character*70 HDR(25)
      common/HDR/HDR
 
      FILEU = FILE
      do i = 75,2,-1
         if (FILE(i:i+4).eq.'.fits') FILEU = FILE(1:i+4)
         enddo
 
      open(10,file=FILEU,status='unknown',
     .     err=900,recl=2880,form='UNFORMATTED',
     .     access='DIRECT')
 
      do k = 00, 34
         write(buffc(k*80+1:k*80+80),'(80x)')
         enddo
 
      write(10,rec=i,iostat=ios) buffc
 
      write(buffc( 0*80+1: 1*80),'(''SIMPLE  =                    T'')')
      write(buffc( 1*80+1: 2*80),'(''BITPIX  =                  -32'')')
      write(buffc( 2*80+1: 3*80),'(''NAXIS   = '',8x,i12)') 2
      write(buffc( 3*80+1: 4*80),'(''NAXIS1  = '',8x,i12)') PXDIMX
      write(buffc( 4*80+1: 5*80),'(''NAXIS2  = '',8x,i12)') PXDIMY
      write(buffc( 5*80+1: 6*80),'(''DATATYPE= '',9a)')
     .                          " 'REAL*4' "
      write(buffc(07*80+1:08*80),'(''COMMENT   '',a05)') '     '
      write(buffc(08*80+1:09*80),'(''COMMENT   '',a05)') '     '
      write(buffc(09*80+1:10*80),'(''CRPIX1  = '',a70)') HDR(01)
      write(buffc(10*80+1:11*80),'(''CRPIX2  = '',a70)') HDR(02)
      write(buffc(11*80+1:12*80),'(''CRVAL1  = '',a70)') HDR(03)
      write(buffc(12*80+1:13*80),'(''CRVAL2  = '',a70)') HDR(04)
      write(buffc(13*80+1:14*80),'(''CTYPE1  = '',a70)') HDR(05)
      write(buffc(14*80+1:15*80),'(''CTYPE2  = '',a70)') HDR(06)
      write(buffc(15*80+1:16*80),'(''CD1_1   = '',a70)') HDR(07)
      write(buffc(16*80+1:17*80),'(''CD1_2   = '',a70)') HDR(08)
      write(buffc(17*80+1:18*80),'(''CD2_1   = '',a70)') HDR(09)
      write(buffc(18*80+1:19*80),'(''CD2_2   = '',a70)') HDR(10)
      write(buffc(19*80+1:20*80),'(''ORIENTAT= '',a70)') HDR(11)
      write(buffc(20*80+1:21*80),'(''PA_APER = '',a70)') HDR(12)
      write(buffc(21*80+1:22*80),'(''PA_V3   = '',a70)') HDR(13)
      write(buffc(22*80+1:23*80),'(''DATE-OBS= '',a70)') HDR(14)
      write(buffc(23*80+1:24*80),'(''TIME-OBS= '',a70)') HDR(15)
      write(buffc(24*80+1:25*80),'(''EXPTIME = '',a70)') HDR(16)
      write(buffc(25*80+1:26*80),'(''ROOTNAME= '',a70)') HDR(17)
      write(buffc(26*80+1:27*80),'(''TARGNAME= '',a70)') HDR(18)
      write(buffc(27*80+1:28*80),'(''RA_TARG = '',a70)') HDR(19)
      write(buffc(28*80+1:29*80),'(''DEC_TARG= '',a70)') HDR(20)
      write(buffc(29*80+1:30*80),'(''PROPOSID= '',a70)') HDR(21)
      write(buffc(30*80+1:31*80),'(''FILTER1 = '',a70)') HDR(22)
      write(buffc(31*80+1:32*80),'(''FILTER2 = '',a70)') HDR(23)
      write(buffc(33*80+1:34*80),'(''VAFACTOR= '',a70)') HDR(24)
      write(buffc(32*80+1:33*80),'(''CCDGAIN = '',a70)') HDR(25)
      write(buffc(33*80+1:34*80),'(''COMMENT   '',a05)') '     '
      write(buffc(34*80+1:35*80),'(''COMMENT   '',a05)') '     '
      write(buffc(35*80+1:36*80),'(''END       '')')
 
      write(10,rec=i,iostat=ios) buffc
 
 
      ifirst = i+1
      i1 = i
      i2 = i
 
      nbper  = 4*PXDIMX*PXDIMY
      npt    =   PXDIMX*PXDIMY
      nbyte1 = 1
      nbyte2 = nbper
      i1 = i+1 + nbyte1/2880
      i2 = i+1 + nbyte2/2880
 
      do i = i1, i2, 1
         nbyte0 = (i-ifirst)*2880+   1
         nbyteE = (i-ifirst)*2880+2880
         np1 = (nbyte0-nbyte1)/4 + 1
         np2 = (nbyteE-nbyte1)/4 + 1
         call pix2buff_r4(buffb,pix,np1,npt)
         write(10,rec=i,iostat=ios) buffc
         enddo
 
      close(10)
 
      return
 
 900  continue
      print*,'writfits_r4.f ERROR'
      print*,'   FILEU: ',FILEU
      stop
 
      end
 
c-------------------------------------------------------
c
c
      subroutine pix2buff_r4(buff,pix,n1,nt)
      implicit none
      byte buff(2880)
      real*4 pix(*)
      integer n1,nt
 
      byte b(4)
      real*4 r
      equivalence(r,b)
 
      integer i, npu, nbu
      logical islinux
 
      do i = 1, 720
         npu = n1+i-1
         nbu = (i-1)*4
         if (npu.ge.1.and.npu.le.nt) r = pix(npu)
         if (.not.islinux()) then
            buff(nbu+1) = b(1)
            buff(nbu+2) = b(2)
            buff(nbu+3) = b(3)
            buff(nbu+4) = b(4)
            endif
         if (islinux()) then
            buff(nbu+1) = b(4)
            buff(nbu+2) = b(3)
            buff(nbu+3) = b(2)
            buff(nbu+4) = b(1)
            endif
         enddo
 
      return
      end
 
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/FITSIO/GEN/writfits_b1.f"
c****                                         
c*********************************************
                                              
 
 
      subroutine writfits_b1(FILE,pix,PXDIMX,PXDIMY)
      implicit none
 
      character*(*) FILE
      integer PXDIMX,PXDIMY
      byte pix(PXDIMX,PXDIMY)
 
      integer nbyte0
      integer nbyteE
      integer nbyte1
      integer nbyte2
      integer nbper
      integer i,ios
 
      character*2880 buffc
      byte buffb(2880)
      equivalence (buffc,buffb)
 
      integer ifirst, i1, i2
 
      integer np1, np2, npt
 
      character*70 HDR(25)
      common/HDR/HDR
 
      character*80 FILEU
 
      FILEU = FILE
      do i = 75,2,-1
         if (FILE(i:i+4).eq.'.fits') FILEU = FILE(1:i+4)
         enddo
 
 
 
c     write(*,'(''ENTER   writfits_b1: '',80a)') FILEU
 
      open(10,file=FILEU,
     .     status='unknown',
     .     err   =900,
     .     recl  =2880,
     .     form  ='UNFORMATTED',
     .     access='DIRECT')
 
 
      i = 1
      write(buffc( 0*80+1: 1*80),'(''SIMPLE  = T    '')')
      write(buffc( 1*80+1: 2*80),'(''BITPIX  =  8  '')')
      write(buffc( 2*80+1: 3*80),'(''NAXIS   = '',i12)') 2
      write(buffc( 3*80+1: 4*80),'(''NAXIS1  = '',i12)') PXDIMX
      write(buffc( 4*80+1: 5*80),'(''NAXIS2  = '',i12)') PXDIMY
      write(buffc( 5*80+1: 6*80),'(''DATATYPE= '',9a)')
     .                          ' ''BYTE    '''
      write(buffc( 6*80+1: 7*80),'(''DATE    = '',11a)')
     .                          ' ''28/01/00'''
      write(buffc( 7*80+1: 8*80),'(''BSCALE  = '',i12)') 00001
      write(buffc( 8*80+1: 9*80),'(''BZERO   = '',i12)') 00000
      write(buffc(09*80+1:10*80),'(''CRPIX1  = '',a20)') HDR(01)
      write(buffc(10*80+1:11*80),'(''CRPIX2  = '',a20)') HDR(02)
      write(buffc(11*80+1:12*80),'(''CRVAL1  = '',a20)') HDR(03)
      write(buffc(12*80+1:13*80),'(''CRVAL2  = '',a20)') HDR(04)
      write(buffc(13*80+1:14*80),'(''CTYPE1  = '',a20)') HDR(05)
      write(buffc(14*80+1:15*80),'(''CTYPE2  = '',a20)') HDR(06)
      write(buffc(15*80+1:16*80),'(''CD1_1   = '',a20)') HDR(07)
      write(buffc(16*80+1:17*80),'(''CD1_2   = '',a20)') HDR(08)
      write(buffc(17*80+1:18*80),'(''CD2_1   = '',a20)') HDR(09)
      write(buffc(18*80+1:19*80),'(''CD2_2   = '',a20)') HDR(10)
      write(buffc(19*80+1:20*80),'(''ORIENTAT= '',a20)') HDR(11)
      write(buffc(20*80+1:21*80),'(''PA_APER = '',a20)') HDR(12)
      write(buffc(21*80+1:22*80),'(''PA_V3   = '',a20)') HDR(13)
      write(buffc(22*80+1:23*80),'(''DATE-OBS= '',a20)') HDR(14)
      write(buffc(23*80+1:24*80),'(''TIME-OBS= '',a20)') HDR(15)
      write(buffc(24*80+1:25*80),'(''EXPTIME = '',a20)') HDR(16)
      write(buffc(25*80+1:26*80),'(''ROOTNAME= '',a20)') HDR(17)
      write(buffc(26*80+1:27*80),'(''TARGNAME= '',a20)') HDR(18)
      write(buffc(27*80+1:28*80),'(''RA_TARG = '',a20)') HDR(19)
      write(buffc(28*80+1:29*80),'(''DEC_TARG= '',a20)') HDR(20)
      write(buffc(29*80+1:30*80),'(''PROPOSID= '',a20)') HDR(21)
      write(buffc(30*80+1:31*80),'(''FILTER1 = '',a20)') HDR(22)
      write(buffc(31*80+1:32*80),'(''FILTER2 = '',a20)') HDR(23)
      write(buffc(32*80+1:33*80),'(''VAFACTOR= '',a20)') HDR(24)
      write(buffc(33*80+1:34*80),'(''COMMENT   '')')
      write(buffc(34*80+1:35*80),'(''COMMENT   '')')
      write(buffc(35*80+1:36*80),'(''END       '')')
 
c     print*,buffc
      write(10,rec=i,iostat=ios) buffc
 
      ifirst = i+1
      i1 = i
      i2 = i
 
      nbper  =   PXDIMX*PXDIMY
      npt    =   PXDIMX*PXDIMY
      nbyte1 = 1
      nbyte2 = nbper
      i1 = i+1 + nbyte1/2880
      i2 = i+1 + nbyte2/2880
 
 
c      print*,'nbyte1: ',nbyte1
c      print*,'nbyte2: ',nbyte1
c      print*,'    i1: ',i1
c      print*,'    i2: ',i2
      do i = i1, i2, 1
         nbyte0 = (i-ifirst)*2880+   1
         nbyteE = (i-ifirst)*2880+2880
         np1 = (nbyte0-nbyte1) + 1
         np2 = (nbyteE-nbyte1)+ 1
        !if (i.lt.0010) print*,'i: ',i,i1,i2,np1,np2,nbyte0
        !if (i.gt.2900) print*,'i: ',i,i1,i2,np1,np2,nbyte0
         call pix2buff_b1(pix,buffb,np1,npt)
         write(10,rec=i,iostat=ios) buffc
         enddo
 
      close(10)
 
      return
 
 900  continue
      print*,'WRITFITS.f ERROR'
      stop
 
      end
 
      subroutine pix2buff_b1(pix,buffb,n1,nt)
      implicit none
      byte pix(1)
      byte buffb(2880)
 
      integer   n1,nt
 
      integer i
      integer npu
 
      integer ipval
 
      do i = 1, 2880
         npu = (1+(n1-1)*2)+i-1
         npu = (1+(n1-1)*1)+i-1
         if (npu.ge.1.and.npu.le.nt) then
            ipval = pix(npu)
            buffb(i) = ipval
            endif
         enddo
 
      return
      end
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/FITSIO/GEN/writfits_i4.f"
c****                                         
c*********************************************
                                              
 
c------------------------------------------------------
c
c this routine will write an integer*4 array into
c a 2-dimensional fits image
c
      subroutine writfits_i4(FILE,pix,PXDIMX,PXDIMY)
      implicit none
 
      integer PXDIMX,PXDIMY
      character*(*) FILE
      integer*4 pix(PXDIMX,PXDIMY)
 
      integer nbyte0
      integer nbyteE
      integer nbyte1
      integer nbyte2
      integer nbper
      integer i,ios
 
      integer ii, jj
      character*2880 buffc
      byte buffb(2880)
      equivalence (buffc,buffb)
 
      integer ifirst, i1, i2
 
      integer np1, np2, npt
 
      logical DIAG
      data DIAG/.false./
 
      character*70 HDR(25)
      common/HDR/HDR
      character*80 FILEU
 
      FILEU = FILE
      do i = 75,2,-1
         if (FILE(i:i+4).eq.'.fits') FILEU = FILE(1:i+4)
         enddo
 
      open(10,file=FILEU,
     .     status='unknown',
     .     err   =900,
     .     recl  =2880,
     .     form  ='UNFORMATTED',
     .     access='DIRECT')
 
      write(buffc( 0*80+1: 1*80),'(''SIMPLE  = T    '')')
      write(buffc( 1*80+1: 2*80),'(''BITPIX  =  32  '')')
      write(buffc( 2*80+1: 3*80),'(''NAXIS   ='',i12)') 2
      write(buffc( 3*80+1: 4*80),'(''NAXIS1  ='',i12)') PXDIMX
      write(buffc( 4*80+1: 5*80),'(''NAXIS2  ='',i12)') PXDIMY
      write(buffc( 5*80+1: 6*80),'(''DATATYPE='',9a)')
     .                          ' ''INTEGER*4'''
      write(buffc( 6*80+1: 7*80),'(''DATE    ='',11a)')
     .                          ' ''00/00/00'''
      write(buffc( 7*80+1: 8*80),'(''BSCALE  ='',i12)') 00001
      write(buffc( 8*80+1: 9*80),'(''BZERO   ='',i12)') 00000
      write(buffc(09*80+1:10*80),'(''CRPIX1  ='',a20)') HDR(01)
      write(buffc(10*80+1:11*80),'(''CRPIX2  ='',a20)') HDR(02)
      write(buffc(11*80+1:12*80),'(''CRVAL1  ='',a20)') HDR(03)
      write(buffc(12*80+1:13*80),'(''CRVAL2  ='',a20)') HDR(04)
      write(buffc(13*80+1:14*80),'(''CTYPE1  ='',a20)') HDR(05)
      write(buffc(14*80+1:15*80),'(''CTYPE2  ='',a20)') HDR(06)
      write(buffc(15*80+1:16*80),'(''CD1_1   ='',a20)') HDR(07)
      write(buffc(16*80+1:17*80),'(''CD1_2   ='',a20)') HDR(08)
      write(buffc(17*80+1:18*80),'(''CD2_1   ='',a20)') HDR(09)
      write(buffc(18*80+1:19*80),'(''CD2_2   ='',a20)') HDR(10)
      write(buffc(19*80+1:20*80),'(''ORIENTAT='',a20)') HDR(11)
      write(buffc(20*80+1:21*80),'(''PA_APER ='',a20)') HDR(12)
      write(buffc(21*80+1:22*80),'(''PA_V3   ='',a20)') HDR(13)
      write(buffc(22*80+1:23*80),'(''DATE-OBS='',a20)') HDR(14)
      write(buffc(23*80+1:24*80),'(''TIME-OBS='',a20)') HDR(15)
      write(buffc(24*80+1:25*80),'(''EXPTIME ='',a20)') HDR(16)
      write(buffc(25*80+1:26*80),'(''ROOTNAME='',a20)') HDR(17)
      write(buffc(26*80+1:27*80),'(''TARGNAME='',a20)') HDR(18)
      write(buffc(27*80+1:28*80),'(''RA_TARG ='',a20)') HDR(19)
      write(buffc(28*80+1:29*80),'(''DEC_TARG='',a20)') HDR(20)
      write(buffc(29*80+1:30*80),'(''PROPOSID='',a20)') HDR(21)
      write(buffc(30*80+1:31*80),'(''FILTER1 ='',a20)') HDR(22)
      write(buffc(31*80+1:32*80),'(''FILTER2 ='',a20)') HDR(23)
      write(buffc(33*80+1:34*80),'(''VAFACTOR='',a20)') HDR(24)
      write(buffc(32*80+1:33*80),'(''CCDGAIN ='',a20)') HDR(25)
      write(buffc(34*80+1:35*80),'(''COMMENT  '')')
      write(buffc(35*80+1:36*80),'(''END      '')')
 
      i = 1
      write(10,rec=i,iostat=ios) buffc
 
      ifirst = i+1
      i1 = i
      i2 = i
 
      nbper  = 4*PXDIMX*PXDIMY
      npt    =   PXDIMX*PXDIMY
      nbyte1 = 1
      nbyte2 = nbper
      i1 = i+1 + nbyte1/2880
      i2 = i+1 + nbyte2/2880
 
      if (DIAG) then
         print*,'----->      i1: ',i1
         print*,'----->      i2: ',i2
         print*,'----->  PXDIMX: ',PXDIMX
         print*,'----->  PXDIMY: ',PXDIMY
         print*,'----->   nbper: ',nbper
         print*,'----->     npt: ',npt
         print*,'----->  C'
         endif
 
      do ii = 1, PXDIMX
      do jj = 1, PXDIMY
         pix(ii,jj) = pix(ii,jj)
         enddo
         enddo
 
      do i = i1, i2, 1
         nbyte0 = (i-ifirst)*2880+   1
         nbyteE = (i-ifirst)*2880+2880
         np1 = (nbyte0-nbyte1)/4 + 1
         np2 = (nbyteE-nbyte1)/4 + 1
         call pix2buff_i4(buffb,pix,np1,npt)
         write(10,rec=i,iostat=ios) buffc
         enddo
 
      close(10)
 
      return
 
 900  continue
      print*,' '
      print*,'WRITFITS_i4 ERROR: '
      print*,' COULD NOT OPEN FILE: '
      print*,' FILE: ',FILEU
      print*,' '
      stop
 
      end
 
 
c
c
c
      subroutine pix2buff_i4(buff,pix,n1,nt)
      implicit none
      byte buff(2880)
      integer*4 pix(*)
      integer n1,nt
 
      byte b(4)
      integer ii
      equivalence(ii,b)
 
      integer i, npu, nbu
      logical islinux
 
      do i = 1, 720
         npu = n1+i-1
         nbu = (i-1)*4
         if (npu.ge.1.and.npu.le.nt) ii = pix(npu)
         if (.not.islinux()) then
            buff(nbu+1) = b(1)
            buff(nbu+2) = b(2)
            buff(nbu+3) = b(3)
            buff(nbu+4) = b(4)
            endif
         if (islinux()) then
            buff(nbu+1) = b(4)
            buff(nbu+2) = b(3)
            buff(nbu+3) = b(2)
            buff(nbu+4) = b(1)
            endif
         enddo
 
      return
      end
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/FITSIO/GEN/readfits_r4_3D.f"
c****                                         
c*********************************************
                                              
 
 
c------------------------------------------------------
c
c
c
      subroutine readfits_r4_3D(FILE,pix,NX,NY,NZ)
      implicit none
 
      character*(*) FILE
      integer NX, NY, NZ
      real*4  pix(NX,NY,NZ)
 
      integer naxes
      integer laxis(3)
 
      character*8  field
      character*20 stream
 
      real    pixu
      integer nbyte0
      integer nbyteE
      integer nbyte1
      integer nbyte2
      integer nbper
      integer i,ios, k
      integer ii, jj, kk
      integer n
      integer NXU, NYU, NZU
 
      character*2880 buffc
      byte buffb(2880)
      equivalence (buffc,buffb)
 
      real*4 rbuff(720)
 
      integer ifirst, i1, i2
      integer j
 
      integer np1, np2, npt
      integer nextend
      integer nread
      real bscale, bzero
      integer bitpix
 
      integer r2i
 
      logical DIAG
      data DIAG /.false./
 
 
      character*80 FILEU
 
      FILEU = FILE
      do i = 75,2,-1
         if (FILE(i:i+4).eq.'.fits') FILEU = FILE(1:i+4)
         enddo
 
 
c      write(*,'(''ENTER readfits_i2: '',80a)') FILEU
 
      open(10,file=FILEU,status='old',
     .     err=900,recl=2880,form='UNFORMATTED',
     .     access='DIRECT')
 
      if (DIAG) print*,'...opened'
 
      bscale = 1
      bzero  = 0
 
      naxes = -1
      laxis(1) = 1
      laxis(2) = 1
      laxis(3) = 1
      nextend = 0
 
      i = 0
      nread = 0
 100  continue
      i = i + 1
      read(10,rec=i,iostat=ios) buffc
      if (DIAG) print*,'READREC: ',i
      do k = 0, 35, 1
         if (DIAG) write(*,'(i4,1x,i4,1x,a80)')
     .                   i,k,buffc(k*80+1:k*80+80)
         field  = buffc(k*80+01:k*80+08)
         stream = buffc(k*80+11:k*80+31)
         if (field.eq.'NAXIS   ') read(stream,*) naxes
         if (field.eq.'NAXIS1  ') read(stream,*) laxis(1)
         if (field.eq.'NAXIS2  ') read(stream,*) laxis(2)
         if (field.eq.'NAXIS3  ') read(stream,*) laxis(3)
         if (field.eq.'NEXTEND ') read(stream,*) nextend
         if (field.eq.'BITPIX  ') read(stream,*) bitpix
         if (field.eq.'BSCALE  ') read(stream,*) bscale
         if (field.eq.'BZERO   ') read(stream,*) bzero
         if (field.eq.'END     ') goto 101
         enddo
      goto 100
 101  continue
 
      nread = nread + 1
      if (DIAG) then
         print*,'----------------------------------------'
         print*,'  NREAD: ',nread
         print*,'NEXTEND: ',nextend
         print*,'  NAXIS: ',naxes
         print*,'  LAXIS: ',laxis(1),laxis(2),laxis(3)
         print*,' BITPIX: ',bitpix
         print*,' BSCALE: ',bscale
         print*,'  BZERO: ',bzero
         endif
 
      ifirst = i+1
      i1 = i
      i2 = i
 
      if (BITPIX.ne.-32) then
         print*,'readfits_r4...: '
         print*,'                '
         print*,'   you called a routine to read in an'
         print*,'   real*4 mage, and the image you'
         print*,'   gave it has BITPIX = ',BITPIX
         print*,'   '
         print*,'   FILEU: ',FILEU
         print*,'   '
         stop
         endif
 
      nbper = 4*laxis(1)*laxis(2)*laxis(3)
      npt   =   laxis(1)*laxis(2)*laxis(3)
      nbyte1 = 1
      nbyte2 = nbper
      i1 = i+1 + nbyte1/2880
      i2 = i+1 + nbyte2/2880
 
      if (laxis(1).ne.NX.or.
     .    laxis(2).ne.NY.or.
     .    laxis(3).ne.NZ) then
         write(*,'(''readfits_r4_3D: '',80a)') FILE
         write(*,'(''readfits_r4_3D: '',80a)') FILEU
         print*,' '
         print*,'  laxis1: ',laxis(1)
         print*,'      NX: ',NX
         print*,' '
         print*,'  laxis2: ',laxis(2)
         print*,'      NY: ',NY
         print*,' '
         print*,'  laxis3: ',laxis(3)
         print*,'      NZ: ',NZ
         print*,' '
         stop
         endif
 
      NXU = laxis(1)
      NYU = laxis(2)
      NZU = laxis(2)
 
      if (DIAG) then
         print*,'     NX: ',NX
         print*,'     NY: ',NY
         print*,'     NZ: ',NZ
         print*,'  NBPER: ',nbper
         print*,'  NBYT1: ',nbyte1
         print*,'  NBYT2: ',nbyte2
         print*,' IFIRST: ',ifirst
         print*,'     I1: ',i1
         print*,'     I2: ',i2
         print*,'    NPT: ',NPT
         endif
 
      do i = i1, i2, 1
         read(10,rec=i,iostat=ios) buffc
         nbyte0 = (i-ifirst)*2880+   1
         nbyteE = (i-ifirst)*2880+2880
         np1 = (nbyte0-nbyte1)/4 + 1
         np2 = (nbyteE-nbyte1)/4 + 1
         np2 = min(np2,npt)
         call buff2pix_r4_3D(buffb,rbuff,0001,0720)
         do n = np1, np2, 1
            kk = n/(NXU*NYU)
            jj = (n-kk*NXU*NYU)/NXU
            ii = n-kk*NXU*NYU-jj*NXU
            pixu = rbuff(n-np1+1)*bscale+bzero
            pix(ii,jj+1,kk+1) = pixu
            enddo
         enddo
 
      if (DIAG) then
         print*,'  NBPER: ',nbper
         print*,'  NBYT1: ',nbyte1
         print*,'  NBYT2: ',nbyte2
         print*,' IFIRST: ',ifirst
         print*,'     I1: ',i1
         print*,'     I2: ',i2
         print*,'    NPT: ',NPT
         endif
 
 
      return
 
 900  continue
      print*,'READFITS_I2_3D ERROR'
      stop
 
      end
 
 
 
      subroutine buff2pix_r4_3D(buff,pix,n1,nt)
      implicit none
      byte    buff(2880)
      real*4  pix(720)
      integer n1,nt
 
      byte   b(4)
      real*4 r
      equivalence(r,b)
 
      integer i, npu, nbu
      logical islinux
 
      do i = 0001, 0720
         npu = n1+i-1
         nbu = (i-1)*4
         if (.not.islinux()) then
            b(1) = buff(nbu+1)
            b(2) = buff(nbu+2)
            b(3) = buff(nbu+3)
            b(4) = buff(nbu+4)
            endif
         if (islinux()) then
            b(4) = buff(nbu+1)
            b(3) = buff(nbu+2)
            b(2) = buff(nbu+3)
            b(1) = buff(nbu+4)
            endif
         if (npu.ge.1.and.npu.le.nt) pix(npu) = r
         enddo
 
      return
      end
 
 
 
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/ROUTINES/WFC.09x10/rpsf_phot.f"
c****                                         
c*********************************************
                                              
 
 
c--------------------------------------
c
c this will evaluate a PSF at a given (dx,dy)
c offset; for the regions within 4 pixels of
c the center, it uses bi-cubic interpolation
c
      real function rpsf_phot(x,y,psf)
      implicit none
      real x, y
      real psf(101,101)
 
 
      real    rx, ry
      integer ix, iy   !     3   4
      real    fx, fy   !    *1*  2
 
      real    dd
 
      real A1, B1, C1, D1, E1, F1, V1
      real A2, B2, C2, D2, E2, F2, V2
      real A3, B3, C3, D3, E3, F3, V3
      real A4, B4, C4, D4, E4, F4, V4
 
 
      rx = 51 + x*4
      ry = 51 + y*4
      ix = int(rx)
      iy = int(ry)
      fx = rx-ix
      fy = ry-iy
 
      dd = sqrt(x**2+y**2)
 
 
      rpsf_phot = 0.
      if (dd.gt.12.0) return
 
      if (dd.gt.4.0) then
         rpsf_phot = (1-fx)*(1-fy)*psf(ix  ,iy  )
     .             + ( fx )*(1-fy)*psf(ix+1,iy  )
     .             + (1-fx)*( fy )*psf(ix  ,iy+1)
     .             + ( fx )*( fy )*psf(ix+1,iy+1)
         return
         endif
 
 
      A1 =  psf(ix  ,iy  )
      B1 = (psf(ix+1,iy  )-psf(ix-1,iy  ))/2
      C1 = (psf(ix  ,iy+1)-psf(ix  ,iy-1))/2
      D1 = (psf(ix+1,iy  )+psf(ix-1,iy  )-2*A1)/2
      F1 = (psf(ix  ,iy+1)+psf(ix  ,iy-1)-2*A1)/2
      E1 = (psf(ix+1,iy+1)-A1)
 
      A2 =  psf(ix+1,iy  )
      B2 = (psf(ix+2,iy  )-psf(ix  ,iy  ))/2
      C2 = (psf(ix+1,iy+1)-psf(ix+1,iy-1))/2
      D2 = (psf(ix+2,iy  )+psf(ix  ,iy  )-2*A2)/2
      F2 = (psf(ix+1,iy+1)+psf(ix+1,iy-1)-2*A2)/2
      E2 =-(psf(ix  ,iy+1)-A2)
 
      A3 =  psf(ix  ,iy+1)
      B3 = (psf(ix+1,iy+1)-psf(ix-1,iy+1))/2
      C3 = (psf(ix  ,iy+2)-psf(ix  ,iy  ))/2
      D3 = (psf(ix+1,iy+1)+psf(ix-1,iy+1)-2*A3)/2
      F3 = (psf(ix  ,iy+2)+psf(ix  ,iy  )-2*A3)/2
      E3 =-(psf(ix+1,iy  )-A3)
 
      A4 =  psf(ix+1,iy+1)
      B4 = (psf(ix+2,iy+1)-psf(ix  ,iy+1))/2
      C4 = (psf(ix+1,iy+2)-psf(ix+1,iy  ))/2
      D4 = (psf(ix+2,iy+1)+psf(ix  ,iy+1)-2*A4)/2
      F4 = (psf(ix+1,iy+2)+psf(ix+1,iy  )-2*A4)/2
      E4 = (psf(ix  ,iy  )-A4)
 
 
      V1 = A1
     .   + B1*( fx )
     .   + C1*( fy )
     .   + D1*( fx )**2
     .   + E1*( fx )*( fy )
     .   + F1*( fy )**2
 
      V2 = A2
     .   + B2*(fx-1)
     .   + C2*( fy )
     .   + D2*(fx-1)**2
     .   + E2*(fx-1)*( fy )
     .   + F2*( fy )**2
 
      V3 = A3
     .   + B3*( fx )
     .   + C3*(fy-1)
     .   + D3*( fx )**2
     .   + E3*( fx )*(fy-1)
     .   + F3*(fy-1)**2
 
      V4 = A4
     .   + B4*(fx-1)
     .   + C4*(fy-1)
     .   + D4*(fx-1)**2
     .   + E4*(fx-1)*(fy-1)
     .   + F4*(fy-1)**2
 
      rpsf_phot = (1-fx)*(1-fy)*V1
     .          + ( fx )*(1-fy)*V2
     .          + (1-fx)*( fy )*V3
     .          + ( fx )*( fy )*V4
 
 
      return
      end
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/ROUTINES/WFC.09x10/zero_fix.f"
c****                                         
c*********************************************
                                              
 
 
      subroutine zero_fix(pix)
      implicit none
      real    pix(4096,4096)
 
      integer hist1(25), hist2(25), hist3(25)
      integer i, j
      integer PM2, PM1, P00, PP1, PP2
      integer EM2, EM1, E00, EP1, EP2
      integer ERR
      real    f
      real    A, B, C
      real    fmin, emin
      real    rand
      integer iif
 
      print*,'          '
      print*,'          '
      print*,'ZERO_FIX: (for antiquated way of storing images)'
      print*,'          '
      print*,'          '
 
      call find_hist(pix,hist1)
 
      do i = 0001, 4096
      do j = 0001, 4096
        !write(*,'(2i5,1x,f10.3)') i,j,pix(i,j)
         if (pix(i,j).lt.-0.9) pix(i,j) = pix(i,j)-1
         enddo
         enddo
 
      call find_hist(pix,hist2)
 
c      print*,' '
c      do h = 01, 25
c         hz = h-13
c         write(*,'(i3,1x,i3,1x,2i10)') h,hz,hist1(h),hist2(h)
c         enddo
c      print*,' '
 
      fmin = 0
      emin = 9e9
c      print*,' '
      do iif = 000, 100, 001
         f = iif*0.01
         PM2 = hist2(11)
         PM1 = hist2(13)*( f )
         P00 = hist2(13)*(1-f)
         PP1 = hist2(14)
         PP2 = hist2(15)
         A = P00
         B = ((PP1-PM1)/2.0 + (PP2-PM2)/4.0)/2.0
         C = (PP2+PM2-2*A)/8.0
 
c-----------------------------------------------------
c         A = -0.08571*PM2
c     .       +0.34286*PM1
c     .       +0.48572*P00
c     .       +0.34286*PP1
c     .       -0.08571*PP2
c         B = -0.20000*PM2
c     .       -0.10000*PM1
c     .       +0.00000*P00
c     .       +0.10000*PP1
c     .       +0.20000*PP2
c         C = +0.71430*PM2
c     .       -0.35710*PM1
c     .       -0.71430*P00
c     .       -0.35710*PP1
c     .       +0.71430*PP2
c         print*,' '
c         write(*,'(30x,3f15.2)') A,B,C
c         A = P00
c         B = ((PP1-PM1)/2.0 + (PP2-PM2)/4.0)/2.0
c         C = (PP2+PM2-2*A)/8.0
c         write(*,'(30x,3f15.2)') A,B,C
c         print*,' '
c-----------------------------------------------------
 
         EM2 = A - 2*B + 4*C
         EM1 = A -   B +   C
         E00 = A
         EP1 = A +   B +   C
         EP2 = A + 2*B + 4*C
         ERR = abs(PM2-EM2)
     .       + abs(PM1-EM1)
     .       + abs(P00-E00)
     .       + abs(PP1-EP1)
     .       + abs(PP2-EP2)
c         write(*,'(f8.4,10x,5i8,10x,5i8,10x,i12)') f,
c     .             PM2,PM1,P00,PP1,PP2,
c     .             EM2,EM1,E00,EP1,EP2,
c     .             ERR
         if (ERR.lt.emin) then
            emin = ERR
            fmin = f
            endif
         enddo
c      print*,' '
c      print*,'fmin: ',fmin
c      print*,' '
 
      do i = 0001, 4096
      do j = 0001, 4096
         if (int(pix(i,j)).eq.0) then
            if (rand().lt.fmin) pix(i,j) = -1
            endif
         enddo
         enddo
 
      call find_hist(pix,hist3)
 
c     print*,' '
c      do h = 01, 25
c        hz = h-13
c        write(*,'(i3,1x,i3,1x,3i10)') h,hz,hist1(h),hist2(h),hist3(h)
c        enddo
c      print*,' '
 
      return
 
      end
 
 
c-----------------------------------
c
c
      subroutine find_hist(pix,hist)
      implicit none
 
      real pix(4096,4096)
      integer hist(25)
 
      integer h
      integer i, j
 
      do h = 01, 25
         hist(h) = 0
         enddo
 
      do i = 0001, 4096
      do j = 0001, 4096
         h = 13 + pix(i,j)
         if (h.ge.01.and.h.le.25) hist(h) = hist(h)+1
         enddo
         enddo
 
      return
      end
 
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/ROUTINES/BARSIG/rbarsigs.f"
c****                                         
c*********************************************
                                              
 
 
      subroutine rbarsigs(xlist,NTOT,bar,sig,NUSE,SIGCLIP)
      implicit none
 
      integer NTOT
      real xlist(NTOT)
      real bar
      real sig
      integer NUSE
      real SIGCLIP
 
      integer n
      real*8    bsum, ssum
      integer nsum, nsumo
      integer NIT
 
 
      nsum = 0
      bar = 0.e0
      sig = 9e9
      do NIT = 1, 20
         nsumo = nsum
         bsum = 0.
         ssum = 0.
         nsum = 0.
         do n = 1, NTOT
            if (abs(xlist(n)-bar).le.SIGCLIP*sig) then
               bsum = bsum + xlist(n)
               ssum = ssum + abs(xlist(n)-bar)
               nsum = nsum + 1
               endif
            enddo
         if (nsum.gt.0) bar = bsum / nsum
         if (nsum.gt.1) sig = ssum/(nsum-1)
         if (nsum.lt.0.35*NTOT.and.NIT.ge.4) goto 1
         if (nsum.eq.nsumo.and.NIT.ge.4) goto 1
         enddo
  1   continue
      NUSE = nsum
      if (nsum.le.1) sig = 0.999
 
      return
      end
 
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/ROUTINES/PROC_SATN/max_contig_NAXIS.f"
c****                                         
c*********************************************
                                              
 
      subroutine max_contig_NAXIS(pixp,pixx,UPR_LIM,NAXIS1,NAXIS2)
      implicit none
 
      integer NAXIS1, NAXIS2
      real    pixp(NAXIS1,NAXIS2)
      real    pixx(NAXIS1,NAXIS2)
      integer i ,j
      integer UPR_LIM
 
      integer jmin
      integer jmax
 
      integer L, Ls, Lo, LL
      integer il(99999)
      integer jl(99999)
      real    pmax
      integer NIT, NCHG
 
      do i = 0001, NAXIS1
      do j = 0001, NAXIS2
         pixx(i,j) = pixp(i,j)
         enddo
         enddo
 
      NIT  = 1
    1 NIT  = NIT + 1
      NCHG = 0
      do j = 0001 + 1, NAXIS2-1,  1
      do i = 0001 + 1, NAXIS1-1,  1
         if (pixx(i,j).gt.UPR_LIM) then
            if (pixx(i,j).lt.pixx(i+1,j  )) then
               pixx(i,j) = pixx(i+1,j  )
               NCHG = NCHG + 1
               endif
            if (pixx(i,j).lt.pixx(i-1,j  )) then
               pixx(i,j) = pixx(i-1,j  )
               NCHG = NCHG + 1
               endif
            if (pixx(i,j).lt.pixx(i  ,j+1)) then
               pixx(i,j) = pixx(i  ,j+1)
               NCHG = NCHG + 1
               endif
            if (pixx(i,j).lt.pixx(i  ,j-1)) then
               pixx(i,j) = pixx(i  ,j-1)
               NCHG = NCHG + 1
               endif
            endif
         enddo
         enddo
      do j = NAXIS2-1,  0001 +1, -1
      do i = NAXIS1-1,  0001 +1, -1
         if (pixx(i,j).gt.UPR_LIM) then
            if (pixx(i,j).lt.pixx(i+1,j  )) then
               pixx(i,j) = pixx(i+1,j  )
               NCHG = NCHG + 1
               endif
            if (pixx(i,j).lt.pixx(i-1,j  )) then
               pixx(i,j) = pixx(i-1,j  )
               NCHG = NCHG + 1
               endif
            if (pixx(i,j).lt.pixx(i  ,j+1)) then
               pixx(i,j) = pixx(i  ,j+1)
               NCHG = NCHG + 1
               endif
            if (pixx(i,j).lt.pixx(i  ,j-1)) then
               pixx(i,j) = pixx(i  ,j-1)
               NCHG = NCHG + 1
               endif
            endif
         enddo
         enddo
      write(*,'(''  NIT: '',i6,1x,i8)') NIT,NCHG
      if (NCHG.gt.0) goto 1
 
      return
      end
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/ROUTINES/PROC_SATN/peak_sat_NAXIS.f"
c****                                         
c*********************************************
                                              
 
c---------------------------------------------------
c
c this routine will go through the image pixr() and
c generate two supplemental images:  pixp() and pixx().
c pixp() gives a sense of how saturated each saturated
c pixel is, and pixx() tells you what the maximum value
c is for a touching saturated pixel ; if pixx.eq.pixp,
c then the pixel is the maximum of its distribution
c
      subroutine peak_sat_NAXIS(pixr,pixp,pixx,pixq,
     .                          UPR_LIM,
     .                          NAXIS1,NAXIS2)
      implicit none
 
      integer  NAXIS1,NAXIS2
      real    pixr(NAXIS1,NAXIS2)
      real    pixp(NAXIS1,NAXIS2)
      real    pixx(NAXIS1,NAXIS2)
      byte    pixq(NAXIS1,NAXIS2)
      integer UPR_LIM
 
      integer i,j
      integer ii,jj
      integer imin, imax
      integer jmin, jmax
      integer r
 
      integer nsat, ntot
      real    ptota
      real    ptotb
      real    ptot
 
      print*,' '
      print*,'ENTER PEAK_SAT...',UPR_LIM
 
      do i = 0001+0005, NAXIS1-5
      if (i.eq.i/1000*1000) print*,'  i: ',i
      do j = 0001+0005, NAXIS2-5
         pixp(i,j) = 0
         jmin =  0001
         jmax = NAXIS2
         if (NAXIS2.eq.4096) then
            if (j.le.2048) then
               jmin = 0001
               jmax = 2048
               endif
            if (j.ge.2049) then
               jmin = 2049
               jmax = 4096
               endif
            endif
         if (pixq(i,j).ne.0) then
            r = 1
            ptota = 0
    3       ptota = ptota + pixr(i,j-r)
            if (pixr(i,j-r).gt.0.75*UPR_LIM.and.j-r-1.gt.jmin) then
                r = r + 1
                goto 3
                endif
            r = 1
            ptotb = 0
    2       ptotb = ptotb + pixr(i,j+r)
            if (pixr(i,j+r).gt.0.75*UPR_LIM.and.j+r+1.lt.jmax) then
                r = r + 1
                goto 2
                endif
            r = 2
    1       continue
            nsat = 0
            ntot = 0
            ptot = 0
            do ii = -r, +r
            do jj = -r, +r
               if (i+r.lt. 0001 .or.
     .             i+r.gt.NAXIS1.or.
     .             j+r.lt.jmin.or.
     .             j+r.gt.jmax) then
                   pixp(i,j) = -750
                   goto 666
                   endif
               if (ii**2+jj**2.le.(r+0.5)**2) then
                  ntot = ntot + 1
                  ptot = ptot + max(pixr(i+ii,j+jj),0.0)
                  if (pixr(i+ii,j+jj).gt.UPR_LIM) nsat = nsat + 1
                  endif
               enddo
               enddo
            if (nsat.gt.0.85*ntot) then
               r = r + 1
               goto 1
               endif
            pixp(i,j) = ptot + ptota + ptotb
            endif
  666    continue
         enddo
         enddo
 
      print*,' '
      print*,'FIND MAX_CONTIG...'
      call max_contig_NAXIS(pixp,pixx,UPR_LIM,NAXIS1,NAXIS2)
      print*,'               '
 
      end
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/ROUTINES/PROC_SATN/sat_phot_NAXIS.f"
c****                                         
c*********************************************
                                              
 
c---------------------------------------------
c
c
c
      subroutine sat_phot_NAXIS(pixr,pixp,pixx,pixq,UPR_LIM,
     .                          NAXIS1,NAXIS2,PSFFILE,
     .                          il,jl,zl,Ls)
      implicit none
 
      real      pixr(NAXIS1,NAXIS2)
      real      pixp(NAXIS1,NAXIS2)
      real      pixx(NAXIS1,NAXIS2)
      byte      pixq(NAXIS1,NAXIS2)
      integer   UPR_LIM
      integer   NAXIS1, NAXIS2
      character*80 PSFFILE
      integer il(*), jl(*)
      real    zl(*)
      integer Ls
 
      real      pixs(NAXIS1,NAXIS2)
 
      integer i
      integer j
      integer NARGs, NARG, iargc
 
      logical trip
      character*80 FILEI
      character*80 FILEO
 
 
      integer iu
      real pixe
      integer NSATs
      integer NCENs
      real sr, mbar_sky_NAXIS
      real zr, zrmax
      real apphot_sat2, rtot
      integer nr, nr0
      integer ir
 
      real    reff
      integer ireff
      integer ii, jj
      integer ncirc
      real    rpsf_phot
 
      integer ntot, nap
      real*8 ptot
      real*8 ftot
      real pixpu
      integer iii, jjj
 
      real psfloc(101,101)
 
      logical DIAG
      common /DIAG_/DIAG
 
      if (NAXIS2.eq.4096.and.NAXIS1.eq.4096) then
         do i = 0001, NAXIS1
            pixr(i,2048) = -750
            pixr(i,2049) = -750
            enddo
         endif
 
      do i = 0001, NAXIS1
      do j = 0001, NAXIS2
         pixs(i,j) = pixr(i,j)
         enddo
         enddo
 
      if (DIAG) open(92,file='LOG.SAT_PHOT.ijpl',status='unknown')
      Ls = 0
      do i = 0001, NAXIS1
      do j = 0001, NAXIS2
         if (pixp(i,j).gt.UPR_LIM.and.
     .       pixp(i,j).eq.pixx(i,j)) then
             Ls = Ls + 1
             if (DIAG) write(92,*) i,j,pixp(i,j),Ls
             endif
         enddo
         enddo
      if (DIAG) close(92)
 
      if (DIAG) then
         open(55,file='LOG.peaksat_CEN.reg',status='unknown')
         write(55,'(''# Region file format: DS9 version 3.0'')')
         write(55,'(''global color=yellow'')')
         endif
 
      zrmax = 0.
      print*,'Do sat_phot apphot...'
      Ls = 0
      if (DIAG) open(93,file='LOG.SAT_PHOT.ijms',status='unknown')
      do i = 0001+10, NAXIS1-10
      do j = 0001+10, NAXIS2-10
         if (pixp(i,j).gt.UPR_LIM) then
             if (pixp(i,j).lt.pixx(i,j)) goto 47
c            do ii = -7, 7
c            do jj = -7, 7
c               if (ii**2+jj**2.le.7.5**2.and.
c    .              pixp(i,j).lt.pixp(i+ii,j+jj)) goto 47
c               enddo
c               enddo
             sr = mbar_sky_NAXIS(i,j,25,30,pixs,NAXIS1,NAXIS2)
             call locpsfij_stdpsf(i,j,psfloc,PSFFILE)
             pixpu = pixp(i,j)
             nap  = 0.
             ntot = 0.
             ptot = 0.
             ftot = 0.
             do iii = max(i-50,0001+2), min(i+50,NAXIS1-2)
             do jjj = 0001+2, NAXIS2-2
                if (pixx(iii  ,jjj  ).eq.pixpu.or.
     .              pixx(iii+1,jjj  ).eq.pixpu.or.
     .              pixx(iii-1,jjj  ).eq.pixpu.or.
     .              pixx(iii  ,jjj+1).eq.pixpu.or.
     .              pixx(iii  ,jjj-1).eq.pixpu.or.
     .              pixx(iii+1,jjj+1).eq.pixpu.or.
     .              pixx(iii+1,jjj-1).eq.pixpu.or.
     .              pixx(iii-1,jjj+1).eq.pixpu.or.
     .              pixx(iii-1,jjj-1).eq.pixpu.or.
     .              pixx(iii-2,jjj-2).eq.pixpu.or.
     .              pixx(iii-2,jjj-1).eq.pixpu.or.
     .              pixx(iii-2,jjj  ).eq.pixpu.or.
     .              pixx(iii-2,jjj+1).eq.pixpu.or.
     .              pixx(iii-2,jjj+2).eq.pixpu.or.
     .              pixx(iii-1,jjj+2).eq.pixpu.or.
     .              pixx(iii  ,jjj+2).eq.pixpu.or.
     .              pixx(iii+1,jjj+2).eq.pixpu.or.
     .              pixx(iii+2,jjj+2).eq.pixpu.or.
     .              pixx(iii+2,jjj+1).eq.pixpu.or.
     .              pixx(iii+2,jjj  ).eq.pixpu.or.
     .              pixx(iii+2,jjj-1).eq.pixpu.or.
     .              pixx(iii+2,jjj-2).eq.pixpu.or.
     .              pixx(iii+1,jjj-2).eq.pixpu.or.
     .              pixx(iii  ,jjj-2).eq.pixpu.or.
     .              pixx(iii-1,jjj-2).eq.pixpu) then
                    ptot = ptot + (pixr(iii,jjj)-sr)
                    ftot = ftot + rpsf_phot(1.*(iii-i),
     .                                      1.*(jjj-j),psfloc)
                    nap  = nap  + 1
                    if (pixr(iii,jjj).gt.UPR_LIM) ntot = ntot + 1
                    endif
                enddo
                enddo
             if (ftot.le.0) ftot = 1.0
             rtot = ftot
             zr = ptot/ftot
             Ls = Ls + 1
             if ((Ls.le.0010).or.
     .           (zr.gt.zrmax).or.
     .           (Ls.eq.Ls/0100*0100))
     .            write( *,888) i,j,-2.5*log10(max(zr,1.)),
     .                          sr,zr,ntot,nap,Ls,ftot,
     .                          pixp(i,j), pixx(i,j)
             if (zr.gt.zrmax) zrmax = zr
             if (DIAG) write(93,888) i,j,-2.5*log10(max(zr,1.0)),
     .                               sr,zr,ntot,nap,Ls,ftot,
     .                               pixp(i,j), pixx(i,j)
  888        format(i4.4,1x,i4.4,1x,f10.4,1x,f8.2,
     .                   3x,f12.1,1x,i8,1x,i8,1x,i5.5,1x,f10.6,
     .                   3x,f12.1,1x,f12.1)
             pixr(i,j) = zr
             il(Ls) = i
             jl(Ls) = j
             zl(Ls) = zr
             if (DIAG) write(55,112) i,j,0.5,'blue'
  112        format('image;circle( ',i4,',',i4,',',f6.3,') # color=',a7)
             endif
   47    continue
         enddo
         enddo
      if (DIAG) close(93)
      if (DIAG) close(55)
 
      return
      end
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/ROUTINES/PROC_SATN/max_contig_drz.f"
c****                                         
c*********************************************
                                              
 
c--------------------------------------------------------
c
c
c
c
      subroutine max_contig_drz(pixp,pixx,pixu,NX,NY)
      implicit none
 
      integer NX, NY
      real    pixp(NX,NY)
      real    pixx(NX,NY)
      byte    pixu(NX,NY)
      integer i ,j
 
      integer jmin
      integer jmax
 
      real    pmax
      integer NIT, NCHG
 
      do i = 0001, NX
      do j = 0001, NY
         pixx(i,j) = pixp(i,j)
         enddo
         enddo
 
      NIT  = 1
    1 NIT  = NIT + 1
      NCHG = 0
      do j = 0002, NY-1,  1
      do i = 0002, NX-1,  1
         if (pixu(i,j).eq.1) then
            if (pixx(i,j).lt.pixx(i+1,j  )) then
               pixx(i,j) = pixx(i+1,j  )
               NCHG = NCHG + 1
               endif
            if (pixx(i,j).lt.pixx(i-1,j  )) then
               pixx(i,j) = pixx(i-1,j  )
               NCHG = NCHG + 1
               endif
            if (pixx(i,j).lt.pixx(i  ,j+1)) then
               pixx(i,j) = pixx(i  ,j+1)
               NCHG = NCHG + 1
               endif
            if (pixx(i,j).lt.pixx(i  ,j-1)) then
               pixx(i,j) = pixx(i  ,j-1)
               NCHG = NCHG + 1
               endif
            endif
         enddo
         enddo
      do j = NY-1, 0002, -1
      do i = 0002, NX-1,  1
         if (pixu(i,j).eq.1) then
            if (pixx(i,j).lt.pixx(i+1,j  )) then
               pixx(i,j) = pixx(i+1,j  )
               NCHG = NCHG + 1
               endif
            if (pixx(i,j).lt.pixx(i-1,j  )) then
               pixx(i,j) = pixx(i-1,j  )
               NCHG = NCHG + 1
               endif
            if (pixx(i,j).lt.pixx(i  ,j+1)) then
               pixx(i,j) = pixx(i  ,j+1)
               NCHG = NCHG + 1
               endif
            if (pixx(i,j).lt.pixx(i  ,j-1)) then
               pixx(i,j) = pixx(i  ,j-1)
               NCHG = NCHG + 1
               endif
            endif
         enddo
         enddo
      write(*,'(''  NIT: '',i6,1x,i8)') NIT,NCHG
      if (NCHG.gt.0) goto 1
 
      return
      end
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/ROUTINES/PROC_SATN/peak_sat_drz.f"
c****                                         
c*********************************************
                                              
 
c---------------------------------------------------
c
c this routine will go through the image pixr() and
c generate two supplemental images:  pixp() and pixx().
c pixp() gives a sense of how saturated each saturated
c pixel is, and pixx() tells you what the maximum value
c is for a touching saturated pixel ; if pixx.eq.pixp,
c then the pixel is the maximum of its distribution
c
      subroutine peak_sat_drz(pixr,pixu,pixp,pixx,NX,NY)
      implicit none
 
      integer NX, NY
      real pixr(NX,NY)
      byte pixu(NX,NY)
      real pixp(NX,NY)
      real pixx(NX,NY)
 
      integer i,j
      integer ii,jj
      integer imin, imax
      integer jmin, jmax
      integer r
 
      integer nsat, ntot
      real    ptota
      real    ptotb
      real    ptot
 
      print*,'ENTER PEAK_SAT...'
 
      do i = 00001+0005, NX-0005
      if (i.eq.i/100*100) print*,'  i: ',i
      do j = 00001+0005, NY-0005
         pixp(i,j) = 0
         if (pixu(i,j).eq.1) then
            r = 2
    1       continue
            nsat = 0
            ntot = 0
            ptot = 0
            do ii = -r, +r
            do jj = -r, +r
               if (i+r.lt.0001.or.
     .             i+r.gt.4096.or.
     .             j+r.lt.jmin.or.
     .             j+r.gt.jmax) then
                   pixp(i,j) = -750
                   goto 666
                   endif
               if (ii**2+jj**2.le.(r+0.5)**2) then
                  ntot = ntot + 1
                  ptot = ptot + max(pixr(i+ii,j+jj),0.0)
                  if (pixu(i+ii,j+jj).eq.1) nsat = nsat + 1
                  endif
               enddo
               enddo
            if (nsat.gt.0.85*ntot) then
               r = r + 1
               goto 1
               endif
            pixp(i,j) = ptot + ptota + ptotb
            endif
  666    continue
         enddo
         enddo
 
      print*,'               '
      print*,'FIND MAX_CONTIG_DRZ...'
      print*,'---> go through the image pixel by pixel and determine'
      print*,'     the brightest pixp pixel that is saturatedly '
      print*,'     contiguous'
      print*,'               '
      call max_contig_drz(pixp,pixx,pixu,NX,NY)
 
      print*,'               '
      print*,'               '
      end
 
 
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/WCS/rd2x.f"
c****                                         
c*********************************************
                                              
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
      real*8 xrad
      real*8 x, y, z
      real*8 xx,yy,zz
 
      cosra = cos((r-r0)*(_PI_)/180.0d0)
      sinra = sin((r-r0)*(_PI_)/180.0d0)
 
      cosde = cos(d *(_PI_)/180.0d0)
      sinde = sin(d *(_PI_)/180.0d0)
      cosd0 = cos(d0*(_PI_)/180.0d0)
      sind0 = sin(d0*(_PI_)/180.0d0)
 
      rrrr = sind0*sinde + cosd0*cosde*cosra
 
 
      xrad = cosde*sinra/rrrr
      rd2x = xrad*180.0d0/_PI_
 
      x  = cosde*cos(r *(_PI_)/180.0d0)
      y  = cosde*sin(r *(_PI_)/180.0d0)
      z  = sinde
      xx = cosd0*cos(r0*(_PI_)/180.0d0)
      yy = cosd0*sin(r0*(_PI_)/180.0d0)
      zz = sind0
 
      if (x*xx + y*yy + z*zz.lt.0) rd2x = 90.0
 
 
      return
      end
 
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/WCS/rd2y.f"
c****                                         
c*********************************************
                                              
 
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
      real*8 yrad
      real*8 x, y, z
      real*8 xx,yy,zz
 
      cosra = cos((r-r0)*(_PI_)/180.0d0)
      sinra = sin((r-r0)*(_PI_)/180.0d0)
      cosde = cos(d *(_PI_)/180.0d0)
      sinde = sin(d *(_PI_)/180.0d0)
      cosd0 = cos(d0*(_PI_)/180.0d0)
      sind0 = sin(d0*(_PI_)/180.0d0)
 
      rrrr = sind0*sinde + cosd0*cosde*cosra
 
      yrad = (cosd0*sinde-sind0*cosde*cosra)/rrrr
      rd2y = yrad*180.0d0/_PI_
 
      x  = cosde*cos(r *(_PI_)/180.0d0)
      y  = cosde*sin(r *(_PI_)/180.0d0)
      z  = sinde
      xx = cosd0*cos(r0*(_PI_)/180.0d0)
      yy = cosd0*sin(r0*(_PI_)/180.0d0)
      zz = sind0
 
      if (x*xx + y*yy + z*zz.lt.0) rd2y = 90
 
      return
      end
 
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/WCS/xy2r.f"
c****                                         
c*********************************************
                                              
 
c-----------------------------------------------
c
c This routine will take a point (x,y) in the
c tangent plane, and the coordinate of the
c tangent-point (ra0,dec0), and will compute
c the ra coordinate that corresponds to (x,y).
c
      real*8 function xy2r(x,y,r0,d0)
      implicit none
 
      real*8 x, y
      real*8 r0,d0
 
      real*8 cosde, sinde
      real*8 cosd0, sind0
      real*8 tandr, dr
      real*8 xrad, yrad
 
      xrad = x/180.0d0*(_PI_)
      yrad = y/180.0d0*(_PI_)
 
      cosd0 = cos(d0*(_PI_)/180.0d0)
      sind0 = sin(d0*(_PI_)/180.0d0)
 
      tandr = xrad/(cosd0-yrad*sind0)
      dr    = atan(tandr)
      xy2r  = r0 + dr*180.0d0/(_PI_)
 
      return
      end
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/WCS/xy2d.f"
c****                                         
c*********************************************
                                              
 
c-----------------------------------------------
c
c This routine will take a point (x,y) in the
c tangent plane, and the coordinate of the
c tangent-point (ra0,dec0), and will compute
c the ra coordinate that corresponds to (x,y).
c
      real*8 function xy2d(x,y,r0,d0)
      implicit none
 
      real*8 x, y
      real*8 r0,d0
 
      real*8 cosd0, sind0
      real*8 tandr, dr
      real*8 cosdr
      real*8 tande
      real*8 xrad, yrad
 
      xrad = x/180.0d0*(_PI_)
      yrad = y/180.0d0*(_PI_)
 
      cosd0 = cos(d0*(_PI_)/180.0d0)
      sind0 = sin(d0*(_PI_)/180.0d0)
 
      tandr = xrad/(cosd0-yrad*sind0)
      dr    = atan(tandr)
 
      cosdr = cos(dr)
 
      tande = cosdr*(sind0+yrad*cosd0)/
     .              (cosd0-yrad*sind0)
 
      !print*,'---> tande: ',tande
      xy2d = atan(tande)*180.0d0/(_PI_)
 
      return
      end
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/WCS/rd2lb_G.f"
c****                                         
c*********************************************
                                              
 
c--------------------------------------
c
c ra,dec to Galactic longitude, latitude
c
      subroutine rd2lb_G(ra,dec,l,b)
      implicit none
 
      real*8 ra, dec
      real*8 l, b
 
      b = 180/3.14159*asin(cos(3.14159/180*dec)
     .                    *cos(3.14159/180*27.4)
     .                    *cos(3.14159/180*(ra-192.25))
     .                    +sin(3.14159/180*dec)
     .                    *sin(3.14159/180*27.4))
 
      l = 180/3.14159*atan2(sin(3.14159/180*dec)
     .                     -sin(3.14159/180*b)
     .                     *sin(3.14159/180*27.4)
     .                     ,cos(3.14159/180*dec)
     .                     *sin(3.14159/180*(ra-192.25))
     .                     *cos(3.14159/180*27.4)) + 33.
 
      if (b.gt.180) b = b - 360
      if (l.lt.000) l = l + 360
 
      return
      end
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/WCS/lb2rd_G.f"
c****                                         
c*********************************************
                                              
 
c--------------------------------------
c
c Galactic longitude, latitude to ra, dec
c
      subroutine lb2rd_G(l,b,ra,dec)
      implicit none
 
      real*8 l, b
      real*8 ra, dec
 
      dec = 180/3.14159*asin(cos(3.14159/180*b)
     .                      *cos(3.14159/180*27.4)
     .                      *sin(3.14159/180*(l-33.0))
     .                      +sin(3.14159/180*b)
     .                      *sin(3.14159/180*27.4))
 
      ra = 180/3.14159*atan2(cos(3.14159/180*b)
     .                      *cos(3.14159/180*(l-33))
     .                      ,sin(3.14159/180*b)
     .                      *cos(3.14159/180*27.4)
     .                      -cos(3.14159/180*b)
     .                      *sin(3.14159/180*27.4)
     .                      *sin(3.14159/180*(l-33))) + 192.25
 
      if (l.lt.0) l = l + 360
 
      return
      end
 
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/WCS/rd2lb_E.f"
c****                                         
c*********************************************
                                              
 
c--------------------------------------
c
c ra,dec to ecliptic longitude, latitude
c
      subroutine rd2lb_E(ra,dec,l,b)
      implicit none
 
      real*8 ra, dec
      real*8 l, b
 
      real*8 pi8
      real*8 ls8
 
      pi8 = 3.1415926535897932
      ls8 = 23.44189
 
 
      b = 180d0/pi8*asin(sin(pi8/180d0*dec)
     .                     *cos(pi8/180d0*ls8)
     .                     -cos(pi8/180d0*dec)
     .                     *sin(pi8/180d0*ls8)
     .                     *sin(pi8/180d0*ra))
 
      l = 180d0/pi8*atan2(sin(pi8/180d0*ra)
     .                     *cos(pi8/180d0*ls8)
     .                     +tan(pi8/180d0*dec)
     .                     *sin(pi8/180d0*ls8)
     .                     ,cos(pi8/180d0*ra))
 
      if (b.gt.180) b = b - 360
      if (l.lt.000) l = l + 360
 
      return
      end
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/WCS/lb2rd_E.f"
c****                                         
c*********************************************
                                              
 
c--------------------------------------
c
c ra,dec to ecliptic longitude, latitude
c
      subroutine lb2rd_E(l,b,ra,dec)
      implicit none
 
      real*8 l, b
      real*8 ra, dec
 
      real*8 pi8
      real*8 ls8
 
      pi8 = 3.1415926535897932
      ls8 = 23.44189
 
      dec = 180d0/pi8*asin(sin(pi8/180d0*ls8)
     .                       *sin(pi8/180d0*l)
     .                       *cos(pi8/180d0*b)
     .                       +cos(pi8/180d0*ls8)
     .                       *sin(pi8/180d0*b))
 
      ra  = 180d0/pi8*atan2(cos(pi8/180d0*ls8)
     .                       *sin(pi8/180d0*l)
     .                       *cos(pi8/180d0*b)
     .                       -sin(pi8/180d0*ls8)
     .                       *sin(pi8/180d0*b)
     .                       ,cos(pi8/180d0*l)
     .                       *cos(pi8/180d0*b))
 
      if (ra.lt.0) ra = ra + 360.0d0
 
      return
      end
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/WCS/string2ra.f"
c****                                         
c*********************************************
                                              
 
c-------------------------------------------
c
c this routine will convert a string into RA,
c it can handle both decimal and sexigesimal
c
      real*8 function string2ra(string)
      implicit none
      character*15 string
 
      logical iscolon
      character*15 stringc
      real rah, ram, ras
      integer i
 
      iscolon = .false.
      do i = 01, 15
         stringc(i:i) = string(i:i)
         if (stringc(i:i).eq.':') then
            iscolon = .true.
            stringc(i:i) = ' '
            endif
         enddo
 
      if (.not.iscolon) then
         read(stringc,*) string2ra
         return
         endif
      if (iscolon) then
         read(stringc,*) RAH, RAM, RAS
         string2ra = (rah + ram/60. + ras/60./60.)*15.0
         return
         endif
 
      return
      end
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/WCS/string2de.f"
c****                                         
c*********************************************
                                              
 
c-------------------------------------------
c
c this routine will convert a string into RA,
c it can handle both decimal and sexigesimal
c
      real*8 function string2de(string)
      implicit none
      character*15 string
 
      logical iscolon
      logical isneg
      character*15 stringc
      real DED, DEM, DES
      integer i
 
 
      iscolon = .false.
      isneg   = .false.
      do i = 01, 15
         stringc(i:i) = string(i:i)
         if (stringc(i:i).eq.':') then
            iscolon = .true.
            stringc(i:i) = ' '
            endif
         if (stringc(i:i).eq.'-') isneg = .true.
         enddo
 
      if (.not.iscolon) then
         read(stringc,*) string2de
         return
         endif
 
      if (iscolon) then
         read(stringc,*) DED, DEM, DES
         string2de = (abs(DED) + DEM/60. + DES/60./60.)
         if (isneg) string2de = -string2de
         return
         endif
 
      return
      end
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/WCS/ra2sexig.f"
c****                                         
c*********************************************
                                              
 
c-------------------------------------------
c
c this routine will convert a string into RA,
c it can handle both decimal and sexigesimal
c
      character*11 function ra2sexig(ra)
      implicit none
      real*8 ra
 
      integer rah, ram, rasi, rasf_x100
      real    ras, rasf
 
      rah  = int(ra/15)
      ram  = int((ra/15-rah)*60)
      ras  = (ra/15-rah-ram/60.)*60*60
      rasi = int(ras)
      rasf = ras-rasi
      rasf_x100 = rasf*100
 
      write(ra2sexig,'(i2.2,'':'',i2.2,'':'',i2.2,''.'',i2.2)')
     .    rah, ram, rasi, rasf_x100
 
      return
      end
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/WCS/de2sexig.f"
c****                                         
c*********************************************
                                              
 
c-------------------------------------------
c
c this routine will convert a string into RA,
c it can handle both decimal and sexigesimal
c
      character*12 function de2sexig(de)
      implicit none
      real*8 de
 
      real    deu
      integer ded, dem, desi, desf_x10
      real    des, desf
      character neg
 
      neg = '+'
      if (de.lt.0) neg = '-'
      deu = abs(de)
      ded = int(deu)
      dem = int((deu-ded)*60)
      des = (deu-ded-dem/60.)*60*60
      desi = int(des)
      desf = des-desi
      desf_x10 = desf*100
 
      write(de2sexig,'(a1,i2.2,'':'',i2.2,'':'',i2.2,''.'',i2.2)')
     .    neg, ded, dem, desi, desf_x10
 
      return
      end
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/WCS/fill_wcs_info.f"
c****                                         
c*********************************************
                                              
 
c------------------------------------------------------------
c
c
c
c
      subroutine fill_wcs_info(WCSMODE,NAME,
     .                         CRPIX1_OUT,CRPIX2_OUT,
     .                         CRVAL1_OUT,CRVAL2_OUT,
     .                          CD1_1_OUT, CD1_2_OUT,
     .                          CD2_1_OUT, CD2_2_OUT,
     .                         RCD1_1_OUT,RCD1_2_OUT,
     .                         RCD2_1_OUT,RCD2_2_OUT)
      implicit none
 
      character*(*) WCSMODE
      character*(*) NAME
      real*8        CRPIX1_OUT, CRPIX2_OUT
      real*8        CRVAL1_OUT, CRVAL2_OUT
      real*8         CD1_1_OUT,  CD1_2_OUT
      real*8         CD2_1_OUT,  CD2_2_OUT
      real*8        RCD1_1_OUT, RCD1_2_OUT
      real*8        RCD2_1_OUT, RCD2_2_OUT
 
      character*80 STRING
 
      character*11 ra_string11, ra2sexig
      character*12 de_string12, de2sexig
 
      integer ira_hr
      integer ira_mn
      real    rra_sc
      integer ide_dg
      integer ide_mn
      real    rde_sc
 
      real*8 rRA, rDE
      real*8 rRA0,rDE0
      integer i, ii
 
      character*20 STREAM
      character*80 WCSFILE
 
      WCSFILE = 'NOT FILLED'
c
c------------------------------------------------------------
c
 
      rRA = CRVAL1_OUT
      rDE = CRVAL2_OUT
 
      CRPIX1_OUT = 14100
      CRPIX2_OUT = 14100
      CRVAL1_OUT =    84.537400D0
      CRVAL2_OUT =   -69.138889D0
      CD1_1_OUT  = -9.010672487D-06
      CD1_2_OUT  = -6.309340800D-06
      CD2_1_OUT  = -6.309340800D-06
      CD2_2_OUT  =  9.010672487D-06
 
      CRPIX1_OUT = -9999.00d0
      CRPIX2_OUT = -9999.00d0
      CRVAL1_OUT = -9999.00d0
      CRVAL2_OUT = -9999.00d0
      CD1_1_OUT  = -9999.00d0
      CD1_2_OUT  = -9999.00d0
      CD2_1_OUT  = -9999.00d0
      CD2_2_OUT  = -9999.00d0
 
      rRA0 = CRVAL1_OUT
      rDE0 = CRVAL2_OUT
 
      print*,'FIND_WCSINFO...'
      print*,'   MODE: ',WCSMODE
      print*,'   rRA : ',rRA , rDE
      print*,'   rRA0: ',rRA0, rDE0
 
      if (WCSMODE(1:4).eq.'FIND') then
         call find_obj(rRA,rDE,NAME,rRA0,rDE0)
         print*,'----> find_obj: '
         print*,'---->      rRA: ',rRA
         print*,'---->      rDE: ',rDE
         print*,'---->     NAME: ',NAME
         print*,'---->     rRA0: ',rRA0
         print*,'---->     rDE0: ',rDE0
         CRPIX1_OUT = 0.00
         CRPIX2_OUT = 0.00
         CRVAL1_OUT = rRA0
         CRVAL2_OUT = rDE0
         CD1_1_OUT  = -0.000013888888889d0
         CD1_2_OUT  =  0.000000000000000d0
         CD2_1_OUT  =  0.000000000000000d0
         CD2_2_OUT  =  0.000013888888889d0
         goto 999
         endif
 
      print*,'what is WCSMODE1? ', WCSMODE(1:4), WCSMODE
      if (WCSMODE(1:4).eq.'AUTO') then
         print*,'----> AUTO: '
         print*,'---->     rRA: ',rRA
         print*,'---->     rDE: ',rDE
         CRPIX1_OUT = 0.00
         CRPIX2_OUT = 0.00
         CRVAL1_OUT = rRA
         CRVAL2_OUT = rDE
         CD1_1_OUT  = -0.000013888888889d0
         CD1_2_OUT  =  0.000000000000000d0
         CD2_1_OUT  =  0.000000000000000d0
         CD2_2_OUT  =  0.000013888888889d0
         goto 999
         endif
 
      print*,'what is WCSMODE2? ', WCSMODE(1:4), WCSMODE
      if (WCSMODE(1:4).eq.'WCS=') then
          NAME = 'BY_WCS_TEXT'
          WCSFILE = WCSMODE(5:80)
          open(19,file=WCSFILE,status='old')
          write(*,'(''WCS2UV FILE: '',a80)') WCSFILE
          write(*,'('' '')')
   23     read(19,'(80a)',end=24) STRING
          write(*,'(''---> '',80a)') STRING
          ii = 0
          do i = 1, 80
             if (STRING(i:i).eq.'=') ii = i
             enddo
          if (ii.eq.0) then
             write(*,'(''WCSMODE: '',80a)') WCSMODE
             write(*,'(''PROBLEM WITH WCSFILE: '',80a)') WCSFILE
             write(*,'(''THERE IS NO = IN THIS STRING: '')')
             write(*,'(''STRING:  '',80a)') STRING
             stop
             endif
          if (STRING(1:10).eq.'CRPIX1_OUT')
     .        read(STRING(ii+1:80),*) CRPIX1_OUT
          if (STRING(1:10).eq.'CRPIX2_OUT')
     .        read(STRING(ii+1:80),*) CRPIX2_OUT
          if (STRING(1:10).eq.'CRVAL1_OUT')
     .        read(STRING(ii+1:80),*) CRVAL1_OUT
          if (STRING(1:10).eq.'CRVAL2_OUT')
     .        read(STRING(ii+1:80),*) CRVAL2_OUT
          if (STRING(1:09).eq.'CD1_1_OUT')
     .        read(STRING(ii+1:80),*) CD1_1_OUT
          if (STRING(1:09).eq.'CD1_2_OUT')
     .        read(STRING(ii+1:80),*) CD1_2_OUT
          if (STRING(1:09).eq.'CD2_1_OUT')
     .        read(STRING(ii+1:80),*) CD2_1_OUT
          if (STRING(1:09).eq.'CD2_2_OUT')
     .        read(STRING(ii+1:80),*) CD2_2_OUT
          goto 23
   24     close(19)
          write(*,'('' '')')
          goto 999
          endif
 
      if (WCSMODE(1:4).eq.'HDR=') then
          NAME = 'BYHDR'
          WCSFILE = WCSMODE(5:80)
          print*,'HDR_IS_FITS: ',WCSFILE(1:40)
          call query_hdre(WCSFILE,'CRPIX1  ',STREAM,-1)
          read(STREAM,*) CRPIX1_OUT
          call query_hdre(WCSFILE,'CRPIX2  ',STREAM,-1)
          read(STREAM,*) CRPIX2_OUT
          call query_hdre(WCSFILE,'CRVAL1  ',STREAM,-1)
          read(STREAM,*) CRVAL1_OUT
          call query_hdre(WCSFILE,'CRVAL2  ',STREAM,-1)
          read(STREAM,*) CRVAL2_OUT
          call query_hdre(WCSFILE,'CD1_1   ',STREAM,-1)
          read(STREAM,*) CD1_1_OUT
          call query_hdre(WCSFILE,'CD1_2   ',STREAM,-1)
          read(STREAM,*) CD1_2_OUT
          call query_hdre(WCSFILE,'CD2_1   ',STREAM,-1)
          read(STREAM,*) CD2_1_OUT
          call query_hdre(WCSFILE,'CD2_2   ',STREAM,-1)
          read(STREAM,*) CD2_2_OUT
          goto 999
          endif
 
      if (WCSMODE(1:4).eq.'RDC=') then
          NAME = 'BYRDC'
          print*,'RDC_OK: ',WCSMODE(1:40)
          if (WCSMODE(05:05).ne.'('.or.
     .        WCSMODE(08:08).ne.':'.or.
     .        WCSMODE(11:11).ne.':'.or.
     .        WCSMODE(16:16).ne.','.or.
     .        WCSMODE(20:20).ne.':'.or.
     .        WCSMODE(23:23).ne.':'.or.
     .        WCSMODE(29:29).ne.')') then
              print*,' '
              print*,'WCSMODE FORMAT PROBLEMS (MUST BE EXACT): '
              write(*,'('' TEMPLATE: '',80a)')
     .                'RDC=(00:23:13.5,+33:44:13.22)'
              write(*,'(''   ACTUAL: '',80a)') WCSMODE
              stop
              endif
          read(WCSMODE(06:07),*) ira_hr
          read(WCSMODE(09:10),*) ira_mn
          read(WCSMODE(12:15),*) rra_sc
          read(WCSMODE(18:19),*) ide_dg
          read(WCSMODE(21:22),*) ide_mn
          read(WCSMODE(24:28),*) rde_sc
          print*,'rde_sc: ',rde_sc
          CRPIX1_OUT = 0.000
          CRPIX2_OUT = 0.000
          CRVAL1_OUT = (ira_hr + ira_mn/60.0D0
     .                         + rra_sc/60.0D0/60.0D0)
     .                  *360.0D0/24.0D0
          CRVAL2_OUT = (ide_dg + ide_mn/60.0D0
     .                         + rde_sc/60.0D0/60.0D0)
          if (WCSMODE(17:17).eq.'-') CRVAL2_OUT = -CRVAL2_OUT
          CD1_1_OUT  = -0.000013888889
          CD1_2_OUT  =  0.000000000000
          CD2_1_OUT  =  0.000000000000
          CD2_2_OUT  =  0.000013888889
          goto 999
          endif
 
      if (WCSMODE(1:6).eq.'RDCD=(') then
          STRING = WCSMODE(7:80)
          do i = 01, 80
             if (STRING(i:i).eq.')') STRING(i:i) = ' '
             enddo
          read(STRING,*) CRVAL1_OUT, CRVAL2_OUT
          CRPIX1_OUT = 0.000
          CRPIX2_OUT = 0.000
          CD1_1_OUT  = -0.000013888889
          CD1_2_OUT  =  0.000000000000
          CD2_1_OUT  =  0.000000000000
          CD2_2_OUT  =  0.000013888889
          goto 999
          endif
 
      print*,' '
      write(*,'(''THE WCS DID NOT GET SPECIFIED ADEQUATELY...'')')
      write(*,'(''WCSMODE: '',80a)') WCSMODE
      write(*,'(''WCSFILE: '',80a)') WCSFILE
      print*,' '
      stop
 
  999 continue
      if (CRPIX1_OUT.lt.-9990.or.CRPIX2_OUT.lt.-9990.or.
     .    CRVAL1_OUT.lt.-9990.or.CRVAL2_OUT.lt.-9990.or.
     .    CD1_1_OUT .lt.-9990.or.CD1_2_OUT .lt.-9990.or.
     .    CD2_1_OUT .lt.-9990.or.CD2_2_OUT .lt.-9990) then
          print*,'INSUFFICIENT SPECIFIED WCS INFO.  NEED ALL 8: '
          print*,'  CRPIX1_OUT: ',CRPIX1_OUT
          print*,'  CRPIX2_OUT: ',CRPIX2_OUT
          print*,'  CRVAL1_OUT: ',CRVAL1_OUT
          print*,'  CRVAL2_OUT: ',CRVAL2_OUT
          print*,'   CD1_1_OUT: ', CD1_1_OUT
          print*,'   CD1_2_OUT: ', CD1_2_OUT
          print*,'   CD2_1_OUT: ', CD2_1_OUT
          print*,'   CD2_2_OUT: ', CD2_2_OUT
          stop
          endif
 
      RCD1_1_OUT= CD2_2_OUT/(CD1_1_OUT*CD2_2_OUT-CD2_1_OUT*CD1_2_OUT)
      RCD1_2_OUT=-CD1_2_OUT/(CD1_1_OUT*CD2_2_OUT-CD2_1_OUT*CD1_2_OUT)
      RCD2_1_OUT=-CD2_1_OUT/(CD1_1_OUT*CD2_2_OUT-CD2_1_OUT*CD1_2_OUT)
      RCD2_2_OUT= CD1_1_OUT/(CD1_1_OUT*CD2_2_OUT-CD2_1_OUT*CD1_2_OUT)
      ra_string11 = ra2sexig(CRVAL1_OUT)
      de_string12 = de2sexig(CRVAL2_OUT)
 
      if (.false.) then
         write(*,'('' '')')
         write(*,'(''INPUT WCS2PIX INFORMATION...'')')
         write(*,'('' CRPIX12:  '',2f20.10)') CRPIX1_OUT, CRPIX2_OUT
         write(*,'('' CRVAL12:  '',2f20.10,5x,a10,1x,a12)')
     .                           CRVAL1_OUT,CRVAL2_OUT,
     .                           ra_string11, de_string12
         write(*,'(''  CD1_12:  '',2f20.10)')  CD1_1_OUT,  CD1_2_OUT
         write(*,'(''  CD2_12:  '',2f20.10)')  CD2_1_OUT,  CD2_2_OUT
         write(*,'('' RCD1_12:  '',2f20.10)') RCD1_1_OUT, RCD1_2_OUT
         write(*,'('' RCD2_12:  '',2f20.10)') RCD2_1_OUT, RCD2_2_OUT
         write(*,'('' '')')
         write(*,'(''PIXEL SCALE --- '',f6.3,'' ARCSEC/PIX '')')
     .            sqrt(-CD1_1_OUT*CD2_2_OUT+CD2_1_OUT*CD1_2_OUT)*60*60
         write(*,'('' '')')
         endif
 
      return
      end
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/WCS/inside_poly.f"
c****                                         
c*********************************************
                                              
 
 
c-----------------------------------------------
c
c reports whether a point is inside a convex
c polygon
c
      logical function inside_poly(x,y,xl,yl,Ns)
      implicit none
 
      real*8 x,y
      integer Ns
      real*8 xl(Ns), yl(Ns)
 
      integer n
      real*8  cross1
      real*8  crossN
 
      inside_poly = .true.
      cross1 = (x-xl(Ns))*(yl(1)-yl(Ns))
     .       - (y-yl(Ns))*(xl(1)-xl(Ns))
      do n = 1, Ns-1
           crossN = (x-xl(N))*(yl(N+1)-yl(N))
     .            - (y-yl(N))*(xl(N+1)-xl(N))
           if (cross1*crossN.le.0) inside_poly = .false.
           enddo
      return
      end
 
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/WCS/find_obj.f"
c****                                         
c*********************************************
                                              
 
c----------------------------------------------------
c
c find the closest object of interest to this
c location; info from HARRIS' on-line catalog
c is contained within data statements.
c
      subroutine find_obj(rRA,rDC,NAME,rRA0,rDC0)
      implicit none
      real*8 rRA , rDC      ! input
      character*11 NAME     ! output (nearest obj)
      real*8 rRA0, rDC0     ! output (locn)
 
      real*8 rRAn, rDCn
      real*8 dRA,  dDC
      real*8 dDD, dmin
      integer N
      character*10 RSTRING
      character*09 DSTRING
 
      integer iRAH, iRAM
      real    rRAS
      integer iDEG, iMIN, iSEC
 
#define _NOBJ_ 191
      character*35 INFO(_NOBJ_)
      common /INFO/INFO ! need to do this in LINUX to preserve
      data INFO /
     .'NGC0104 47T 00 24 05.2 -72 04 51.00',
     .'NGC0288     00 52 47.5 -26 35 24.00',
     .'NGC0362     01 03 14.3 -70 50 54.00',
     .'NGC1261     03 12 15.3 -55 13 01.00',
     .'Pal1        03 33 23.0 +79 34 50.00',
     .'AM1     E01 03 55 02.7 -49 36 52.00',
     .'Eridanus    04 24 44.5 -21 11 13.00',
     .'Pal2        04 46 05.9 +31 22 51.00',
     .'NGC1851     05 14 06.3 -40 02 50.00',
     .'NGC1904 M79 05 24 10.6 -24 31 27.00',
     .'NGC2298     06 48 59.2 -36 00 19.00',
     .'NGC2419     07 38 08.5 +38 52 55.00',
     .'Pyxis       09 07 57.8 -37 13 17.00',
     .'NGC2808     09 12 02.6 -64 51 47.00',
     .'E3          09 20 59.3 -77 16 57.00',
     .'Pal3        10 05 31.4 +00 04 17.00',
     .'NGC3201     10 17 36.8 -46 24 40.00',
     .'Pal4        11 29 16.8 +28 58 25.00',
     .'NGC4147     12 10 06.2 +18 32 31.00',
     .'NGC4372     12 25 45.4 -72 39 33.00',
     .'Rup106      12 38 40.2 -51 09 01.00',
     .'NGC4590 M68 12 39 28.0 -26 44 34.00',
     .'NGC4833     12 59 35.0 -70 52 29.00',
     .'NGC5024 M53 13 12 55.3 +18 10 09.00',
     .'NGC5053     13 16 27.0 +17 41 53.00',
     .'NGC5139 OMC 13 26 45.9 -47 28 37.00',
     .'NGC5272 M03 13 42 11.2 +28 22 32.00',
     .'NGC5286     13 46 26.5 -51 22 24.00',
     .'AM4         13 55 50.1 -27 10 22.00',
     .'NGC5466     14 05 27.3 +28 32 04.00',
     .'NGC5634     14 29 37.3 -05 58 35.00',
     .'NGC5694     14 39 36.5 -26 32 18.00',
     .'IC4499      15 00 18.5 -82 12 49.00',
     .'NGC5824     15 03 58.5 -33 04 04.00',
     .'Pal5        15 16 05.3 -00 06 41.00',
     .'NGC5897     15 17 24.5 -21 00 37.00',
     .'NGC5904 M05 15 18 33.8 +02 04 58.00',
     .'NGC5927     15 28 00.5 -50 40 22.00',
     .'NGC5946     15 35 28.5 -50 39 34.00',
     .'BH176       15 39 07.3 -50 03 02.00',
     .'NGC5986     15 46 03.5 -37 47 10.00',
     .'Lynga7      16 11 03.0 -55 18 52.00',
     .'Pal14       16 11 04.9 +14 57 29.00',
     .'NGC6093 M80 16 17 02.5 -22 58 30.00',
     .'NGC6121 M04 16 23 35.5 -26 31 31.00',
     .'NGC6101     16 25 48.6 -72 12 06.00',
     .'NGC6144     16 27 14.1 -26 01 29.00',
     .'NGC6139     16 27 40.4 -38 50 56.00',
     .'Terzan 3    16 28 40.1 -35 21 13.00',
     .'NGC6171     16 32 31.9 -13 03 13.00',
     .'1636-283    16 39 25.5 -28 23 52.00',
     .'NGC6205 M13 16 41 41.5 +36 27 37.00',
     .'NGC6229     16 46 58.9 +47 31 40.00',
     .'NGC6218 M12 16 47 14.5 -01 56 52.00',
     .'NGC6235     16 53 25.4 -22 10 38.00',
     .'NGC6254 M10 16 57 08.9 -04 05 58.00',
     .'NGC6256     16 59 32.6 -37 07 17.00',
     .'Pal15       17 00 02.4 -00 32 31.00',
     .'NGC6266 M62 17 01 12.8 -30 06 49.00',
     .'NGC6273 M19 17 02 37.8 -26 16 05.00',
     .'NGC6284     17 04 28.8 -24 45 53.00',
     .'NGC6287     17 05 09.4 -22 42 29.00',
     .'NGC6293     17 10 10.2 -26 34 55.00',
     .'NGC6304     17 14 32.1 -29 27 44.00',
     .'NGC6316     17 16 37.3 -28 08 24.00',
     .'NGC6341 M92 17 17 07.3 +43 08 11.00',
     .'NGC6325     17 17 59.2 -23 45 57.00',
     .'NGC6333 M09 17 19 11.8 -18 30 59.00',
     .'NGC6342     17 21 10.2 -19 35 14.00',
     .'NGC6356     17 23 35.0 -17 48 47.00',
     .'NGC6355     17 23 58.6 -26 21 13.00',
     .'NGC6352     17 25 29.2 -48 25 22.00',
     .'IC1257      17 27 08.5 -07 05 35.00',
     .'Terzan2 HP3 17 27 33.1 -30 48 08.00',
     .'NGC6366     17 27 44.3 -05 04 36.00',
     .'Terzan4 HP4 17 30 39.0 -31 35 44.00',
     .'HP1         17 31 05.2 -29 58 54.00',
     .'NGC6362     17 31 54.8 -67 02 53.00',
     .'Liller1     17 33 24.5 -33 23 20.00',
     .'NGC6380 Ton 17 34 28.0 -39 04 09.00',
     .'Terzan1     17 35 47.2 -30 28 54.00',
     .'Ton2        17 36 10.5 -38 33 12.00',
     .'NGC6388     17 36 17.0 -44 44 06.00',
     .'NGC6402 M14 17 37 36.1 -03 14 45.00',
     .'NGC6401     17 38 36.6 -23 54 34.00',
     .'NGC6397     17 40 41.3 -53 40 25.00',
     .'Pal6        17 43 42.2 -26 13 21.00',
     .'NGC6426     17 44 54.7 +03 10 13.00',
     .'Djorg1      17 47 28.3 -33 03 56.00',
     .'Terzan5 Trz 17 48 04.9 -24 46 45.00',
     .'NGC6440     17 48 52.7 -20 21 37.00',
     .'NGC6441     17 50 12.9 -37 03 05.00',
     .'Terzan6 HP5 17 50 46.4 -31 16 31.00',
     .'NGC6453     17 50 51.7 -34 35 57.00',
     .'UKS1        17 54 27.2 -24 08 43.00',
     .'NGC6496     17 59 02.0 -44 15 54.00',
     .'Terzan9     18 01 38.8 -26 50 23.00',
     .'Djorg2      18 01 49.1 -27 49 33.00',
     .'NGC6517     18 01 50.6 -08 57 32.00',
     .'Terzan10    18 02 57.4 -26 04 00.00',
     .'NGC6522     18 03 34.1 -30 02 02.00',
     .'NGC6535     18 03 50.7 -00 17 49.00',
     .'NGC6528     18 04 49.6 -30 03 21.00',
     .'NGC6539     18 04 49.8 -07 35 09.00',
     .'NGC6540 Dj  18 06 08.6 -27 45 55.00',
     .'NGC6544     18 07 20.6 -24 59 51.00',
     .'NGC6541     18 08 02.2 -43 30 00.00',
     .'2MSGC01     18 08 21.8 -19 49 47.00',
     .'ESOSC06     18 09 06.0 -46 25 23.00',
     .'NGC6553     18 09 17.6 -25 54 31.00',
     .'2MSGC02     18 09 36.5 -20 46 44.00',
     .'NGC6558     18 10 17.6 -31 45 50.00',
     .'IC1276  Pal 18 10 44.2 -07 12 27.00',
     .'Terzan12    18 12 15.8 -22 44 31.00',
     .'NGC6569     18 13 38.8 -31 49 37.00',
     .'NGC6584     18 18 37.7 -52 12 54.00',
     .'NGC6624     18 23 40.5 -30 21 40.00',
     .'NGC6626 M28 18 24 32.9 -24 52 12.00',
     .'NGC6638     18 30 56.1 -25 29 51.00',
     .'NGC6637 M69 18 31 23.2 -32 20 53.00',
     .'NGC6642     18 31 54.1 -23 28 31.00',
     .'NGC6652     18 35 45.7 -32 59 25.00',
     .'NGC6656 M22 18 36 24.2 -23 54 12.00',
     .'Pal8        18 41 29.9 -19 49 33.00',
     .'NGC6681 M70 18 43 12.7 -32 17 31.00',
     .'NGC6712     18 53 04.3 -08 42 22.00',
     .'NGC6715 M54 18 55 03.3 -30 28 42.00',
     .'NGC6717 Pal 18 55 06.2 -22 42 03.00',
     .'NGC6723     18 59 33.2 -36 37 54.00',
     .'NGC6749     19 05 15.3 +01 54 03.00',
     .'NGC6752     19 10 52.0 -59 59 05.00',
     .'NGC6760     19 11 12.1 +01 01 50.00',
     .'NGC6779 M56 19 16 35.5 +30 11 05.00',
     .'Terzan7     19 17 43.7 -34 39 27.00',
     .'Pal10       19 18 02.1 +18 34 18.00',
     .'Arp2        19 28 44.1 -30 21 14.00',
     .'NGC6809 M55 19 39 59.4 -30 57 44.00',
     .'Terzan8     19 41 45.0 -34 00 01.00',
     .'Pal11       19 45 14.4 -08 00 26.00',
     .'NGC6838 M71 19 53 46.1 +18 46 42.00',
     .'NGC6864 M75 20 06 04.8 -21 55 17.00',
     .'NGC6934     20 34 11.6 +07 24 15.00',
     .'NGC6981 M72 20 53 27.9 -12 32 13.00',
     .'NGC7006     21 01 29.5 +16 11 15.00',
     .'NGC7078 M15 21 29 58.3 +12 10 01.00',
     .'NGC7089 M02 21 33 29.3 -00 49 23.00',
     .'NGC7099 M30 21 40 22.0 -23 10 45.00',
     .'Pal12       21 46 38.8 -21 15 03.00',
     .'Pal13       23 06 44.4 +12 46 19.00',
     .'NGC7492     23 08 26.7 -15 36 41.00',
     .'ANDROMEDM31 00 42 44.3 +41 16 09.00',
     .'LMC         05 23 34.6 -69 45 22.00', ! for now replace LMC with JWST-CAL
     .'JWST-CALIB  05 21 57.0 -69 29 54.00',
     .'SMC         00 52 42.0 -72 49 00.00',
     .'BPTAU       04 19 15.8 +29 06 27.00',
     .'DFTAU       04 27 02.8 +25 42 22.00',
     .'ORION-PAR   05 35 25.0 -05 30 59.00',
     .'UDF         03 32 39.0 -27 47 29.00',
     .'JWST-CALIB  05 21 57.0 -69 29 54.00',
     .'BULGE-SWEEP 17 58 59.1 -29 12 17.73',
     .'BULGE-STANK 17 54 41.7 -29 49 30.00',
     .'BULGE-OGL29 17 48 15.3 -37 09 01.40',
     .'BULGE-BAADE 18 03 10.2 -29 56 33.60',
     .'TBROWN-DISK 18 59 45.0 -04 26 08.40',
     .'NGC6791     19 20 53.0 +37 46 30.00', ! <---
     .'SDSSJ090744 09 07 44.0 +02 45 06.40',
     .'SDSSJ093320 09 33 20.9 +44 17 05.50',
     .'HE0437-5439 04 38 12.8 -53 33 11.80',
     .'SDSSJ091301 09 13 01.0 +30 51 20.20',
     .'SDSSJ091759 09 17 59.6 +67 22 38.80',
     .'SDSSJ110557 11 05 57.5 +09 34 39.47',
     .'SDSSJ113312 11 33 12.1 +01 08 24.87',
     .'SDSSJ094214 09 42 14.1 +20 03 22.07',
     .'SDSSJ102137 10 21 37.1 -00 52 34.77',
     .'SDSSJ120337 12 03 37.6 +18 02 50.35',
     .'SDSSJ105009 10 50 09.6 +03 15 50.67',
     .'SDSSJ105248 10 52 48.3 -00 01 33.94',
     .'SDSSJ144955 14 49 55.6 +31 03 51.37',
     .'SDSSJ101018 10 10 18.8 +30 20 28.22',
     .'SDSSJ110224 11 02 24.4 +02 50 02.77',
     .'MO9-BLG-195 18 03 01.2 -28 23 26.50',
     .'MO9-BLG-046 17 54 43.4 -35 03 07.50',
     .'MO9-BLG-260 17 58 28.6 -26 50 21.10',
     .'OGL9-BULG-D 17 30 0.00 -29 00 00.00',
     .'OGL9-BULG-E 17 30 0.00 -30 00 00.00',
     .'SDSSJ142001 14 20 01.9 +12 44 04.70',
     .'PLEIADES_01 03 47 17.0 +24 14 50.60',
     .'PLEIADES_02 03 48 12.2 +23 59 07.30',
     .'PLEIADES_03 03 48 37.6 +24 13 11.40',
     .'30__DORADUS 05 38 40.0 -69 06 00.00',
     .'GALACTICCEN 17 45 40.0 -29 00 28.10'/
 
 
      dmin = 30.0       ! 30 arcmin finding radius
      NAME = 'NO ID'
      rRA0 = rRA
      rDC0 = rDC
      do N = 1, _NOBJ_
         RSTRING = INFO(N)(13:22)
         DSTRING = INFO(N)(24:32)
         read(RSTRING,*) iRAH,iRAM,rRAS
         read(DSTRING,*) iDEG,iMIN,iSEC
         rRAn = (iRAH+iRAM/60.+rRAS/60./60.)*360/24  ! in degs
         rDCn = abs(iDEG)+iMIN/60.+iSEC/60./60.      ! in degs
c        if (iDEG.lt.0) rDCn = -rDCn
         if (DSTRING(01:01).eq.'-') rDCn = -rDCn
         dRA = (rRA-rRAn)*60.0*cos(rDC*3.14159/180)  ! in arcmin (W+)
         dDC = (rDC-rDCn)*60.0                       ! in arcmin
         dDD = sqrt(dRA**2+dDC**2)
         if (dDD.lt.dmin) then
            dmin = dDD
            rRA0 = rRAn
            rDC0 = rDCn
            NAME = INFO(N)(01:11)
            endif
         enddo
 
 
      return
      end
 
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/PSFs/PERT/find_psfpert.f"
c****                                         
c*********************************************
                                              
 
c------------------------------------------------------------
c
c this routine will derive a perturbation for a library
c PSF so that it will better fit an image
c
 
      subroutine find_psfpert(pixc,HMB,HIFLAG,NAXIS1,NAXIS2,
     .                        psfpert,NPERTs,
     .                        PSFFILE_USE)
      implicit none
 
      real         pixc(NAXIS1,NAXIS2)
      real         HMB
      integer      HIFLAG
      integer      NAXIS1, NAXIS2
      real         psfpert(101,101,_NPRTMX_,_NPRTMX_)
      integer      NPERTs
      character*80 PSFFILE_USE
 
      integer i, ii, iii
      integer j, jj, jjj
 
      real*8 xr, yr, fr, sr
      real   qr
 
      real*8 xro, yro
 
      real   dx, dy
 
      real    FEST
      integer hobs, fnd_hloc
      integer pobs, fnd_ploc
      real    mbar_sky
      integer L, Ls, Lu
      integer il(99999)
      integer jl(99999)
      real*8  xl(99999)
      real*8  yl(99999)
      real    ml(99999)
      real    zl(99999)
      real    sl(99999)
      real    ql(99999), qbar, qsig
      real    pl(99999)
      integer ul(99999)
      real    rxl(99999)
      real    ryl(99999)
      real    pixl(11,11,99999)
      real    psfl(11,11,99999)
      real    psfq
 
      real    rpsf_phot
      real    psfloc(101,101)
      integer NSs, NSu
      real    dpsf(99999)
 
      real    fx, fy
 
      real   FMIN
      integer NIT
      real    histpert_ij(11,11), histpert_min
      real    psfpertu(101,101,_NPRTMX_,_NPRTMX_)
      real    psfpertv(101,101,_NPRTMX_,_NPRTMX_)
      integer PX, PY
      integer ipsf,jpsf
 
      FMIN = HIFLAG
 
      print*,'                                              '
      print*,'ENTER FIND_PSFPERT:                           '
      print*,'                                              '
      print*,'   FIND THE AVERAGE DIFFERENCE BETWEEN THE    '
      print*,'   INPUT LIBRARY PSF AND THE PSF IN THIS IMAGE'
      print*,'                                              '
      print*,'         HIFLAG: ',HIFLAG
      print*,'           FMIN: ',FMIN
      print*,'            HMB: ',HMB
      print*,'            NIT: ',NIT
      print*,'         NAXIS1: ',NAXIS1
      print*,'         NAXIS2: ',NAXIS2
      print*,'            PSF: ',PSFFILE_USE
      print*,'         NPERTs: ',NPERTs
      print*,'                '
 
      if (NPERTs.gt.11) stop 'NPERTs.gt.11'
 
      do PX = 1, NPERTs
      do PY = 1, NPERTs
         do i = 001, 101
         do j = 001, 101
            psfpertu(i,j,PX,PY) = 0.
            enddo
            enddo
         enddo
         enddo
 
      do PX = 1, NPERTs
      do PY = 1, NPERTs
         ii = 1 + (PX-0.5)*(NAXIS1-1)/NPERTs
         jj = 1 + (PY-0.5)*(NAXIS2-1)/NPERTs
         call locpsfij_stdpsf(ii,jj,psfpertv(1,1,PX,PY),PSFFILE_USE)
         enddo
         enddo
 
      NIT = 0
  3   continue
      NIT = NIT + 1
      print*,'             '
      print*,'FIND_PSFPERT FIND STARS TO USE --- NIT: ',NIT
      print*,'                '
      Ls = 0
      do jj = 10, NAXIS2-10
      do ii = 10, NAXIS1-10
         if (pixc(ii,jj).lt.FMIN/9) goto 222
         if (pixc(ii,jj).gt.HIFLAG) goto 222
         FEST = pixc(ii-1,jj+1) + pixc(ii,jj+1) + pixc(ii+1,jj+1)
     .        + pixc(ii-1,jj  ) + pixc(ii,jj  ) + pixc(ii+1,jj  )
     .        + pixc(ii-1,jj-1) + pixc(ii,jj-1) + pixc(ii+1,jj-1)
     .        - HMB*9
         if (FEST.lt. FMIN) goto 222
         do i = -9, 9
         do j = -9, 9
            if (i**2+j**2.le.9.5**2) then
               if (pixc(ii+i,jj+j).gt.pixc(ii,jj)) goto 222
               endif
            if (i**2+j**2.le.5.5**2) then
               if (pixc(ii+i,jj+j).lt.-50) goto 222
               endif
            enddo
            enddo
         fx = (pixc(ii+1,jj)-pixc(ii-1,jj))/2/
     .        (pixc(ii  ,jj)-min(pixc(ii+1,jj),pixc(ii-1,jj)))
         fy = (pixc(ii,jj+1)-pixc(ii,jj-1))/2/
     .        (pixc(ii,jj  )-min(pixc(ii,jj+1),pixc(ii,jj-1)))
         xr = ii + fx
         yr = jj + fy
         xro = xr
         yro = yr
         sr = HMB
         call locpsfij_stdpsf(ii,jj,psfloc,PSFFILE_USE)
         call find_xyzXX_NAXIS(xr,yr,fr,sr,qr,pixc,psfloc,
     .                         NAXIS1,NAXIS2,0) ! 0 means weight by flux (center focus)
         if (qr.gt.0.5) goto 222  ! this may help prevent CRs from
         Ls = Ls + 1              ! being included
         if (Ls.gt.99999) then
            print*,'  Ls    : ', Ls
            print*,'       ii: ', ii
            print*,'       jj: ', jj
            print*,'  99999: ',99999
            stop
            endif
         il(Ls) = ii
         jl(Ls) = jj
         xl(Ls) = xr
         yl(Ls) = yr
         ml(Ls) = -2.5*log10(max(fr,1.))
         zl(Ls) = fr
         sl(Ls) = sr
         ql(Ls) = qr
         pl(Ls) = pixc(ii,jj)
         ul(Ls) = 1
         rxl(Ls) = 1 + (ii-1.0)/NAXIS1*(NPERTs-1)
         ryl(Ls) = 1 + (jj-1.0)/NAXIS2*(NPERTs-1)
         do i = 01, 11
         do j = 01, 11
            iii = ii+(i-6)
            jjj = jj+(j-6)
            dx  = iii-xl(Ls)
            dy  = jjj-yl(Ls)
            pixl(i,j,Ls) = pixc(iii,jjj) - sl(Ls)
            psfl(i,j,Ls) = rpsf_phot(dx,dy,psfloc)
c           write(81,181) Ls,iii,jjj,dx,dy,
c    .                    pixl(i,j,Ls),psfl(i,j,Ls),fr
c 181       format(1x,i5,4x,i4.4,1x,i4.4,
c    .                   4x,f6.3,1x,f6.3,
c    .                   4x,f8.1,1x,f8.6,1x,f9.1,1x,f8.4,
c    .                   4x,f8.1,1x,f7.3)
            enddo
            enddo
         if ((Ls.lt.00025).or.
     .       (Ls.lt.00100.and.Ls.eq.Ls/0010*0010).or.
     .       (Ls.eq.Ls/0100*0100))
     .   write( *,182) xl(Ls),yl(Ls),ml(Ls),zl(Ls),sl(Ls),
     .                 Ls,il(Ls),jl(Ls),
     .                 psfl(6,6,Ls),ql(Ls),
     .                 pixc(ii,jj)
c        write(82,182) xl(Ls),yl(Ls),ml(Ls),zl(Ls),sl(Ls),
c    .                 Ls,il(Ls),jl(Ls),
c    .                 psfl(6,6,Ls),ql(Ls),
c    .                 pixc(ii,jj)
 182     format(f9.3,1x,f9.3,1x,f8.4,1x,f10.1,1x,f7.2,3x,
     .                 i6,3x,i4,1x,i4,1x,f8.6,1x,f8.6,4x,2f9.1)
 222     continue
         enddo
         enddo
 
      print*,' Ls: ',Ls
      print*,'FMIN: ',FMIN
 
      histpert_min = 9e9
      do i = 1, NPERTs
      do j = 1, NPERTs
         histpert_ij(i,j) = 0
         do L = 1, Ls
            histpert_ij(i,j) = histpert_ij(i,j)
     .                       + max(1.-abs(i-rxl(L)),0.)
     .                       * max(1.-abs(j-ryl(L)),0.)
            enddo
         if (histpert_ij(i,j).lt.histpert_min)
     .       histpert_min = histpert_ij(i,j)
         enddo
         enddo
 
      print*,' '
      write(*,'(2x,1x,11(4x,i2,4x))') (i,i=1,NPERTs)
      print*,' '
      do j = NPERTs, 1, -1
         write(*,'(i2,1x,11f10.1)') j,(histpert_ij(i,j),i=1,NPERTs)
         enddo
      print*,' '
      write(*,'(2x,1x,11(4x,i2,4x))') (i,i=1,NPERTs)
      print*,' '
      print*,'histpert_min: ',histpert_min
      print*,' '
 
      if (histpert_min.lt.10) then
         print*,' '
         print*,'NEED TO HAVE AT LEAST 10 PERT-DONOR STARS'
         print*,'IN EACH QUADRANT'
         print*,' '
         FMIN = FMIN/2.0
         if (FMIN.lt.2500) then
             do PX = 1, NPERTs
             do PY = 1, NPERTs
                do ipsf = 001, 101
                do jpsf = 001, 101
                   psfpert(ipsf,jpsf,PX,PY) = 0
                   enddo
                   enddo
                enddo
                enddo
            return
            endif
         goto 3
         endif
 
      print*,' '
      print*,'call ras2pertpsf...',Ls,NPERTs
      print*,' '
      call ras2pertpsf(pixl,psfl,il,jl,xl,yl,zl,sl,ul,Ls,
     .                 rxl,ryl,psfpertu,psfpertv,NPERTs)
      print*,' '
      print*,'back ras2pertpsf...',psfpertu(51,51,1,1)
      print*,' '
 
      do PX = 1, NPERTs
      do PY = 1, NPERTs
         do ipsf = 001, 101
         do jpsf = 001, 101
            psfpert(ipsf,jpsf,PX,PY) = psfpertu(ipsf,jpsf,PX,PY)
            enddo
            enddo
         enddo
         enddo
 
      return
      end
 
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/PSFs/PERT/output_psfperts.f"
c****                                         
c*********************************************
                                              
 
      subroutine output_psfperts(psfperts,NPERTs,STEM_N,NIMs)
      implicit none
 
      real        psfperts(101,101,_NPRTMX_,_NPRTMX_,_NIMMXP_)
      integer     NPERTs
      integer     NIMs
      character*9 STEM_N(NIMs)
 
      real*4, dimension(:,:), allocatable :: show_psfperts
 
      integer  NDIMx, NDIMy
 
      integer NXs, NX, NXu
      integer NYs, NY, NYu
      integer PX, PY
      integer ipsf,jpsf
 
      integer i, j
      integer NIM
      character*09 NUMSTRING
 
      NXs = 10
      NYs = 1 + (NIMs-1)/10
 
      NDIMx = 57*NPERTs*NXs
      NDIMy = 57*NPERTs*NYs
 
      print*,'ENTER output_psfperts...'
      print*,'      NPERTs: ',NPERTs
      print*,'         NXs: ',NXs,NYs
      print*,'        NIMs: ',NIMs
      print*,'       NDIMx: ',NDIMx,NDIMy
      do NIM = 1, NIMs
         write(*,'('' STEM_N('',i2.2,'')'',a9)') NIM,STEM_N(NIM)
c        do i = 1, 9
c           print*,'---> i: ',i,STEM_N(NIM)(i:i)
c           enddo
         enddo
      print*,' '
 
      allocate(show_psfperts(NDIMx,NDIMy))
 
      do i = 001, NDIMx
      do j = 001, NDIMy
         show_psfperts(i,j) = -0.1
         enddo
         enddo
 
      do NIM = 1, NIMs
         NX = NIM - (NIM-1)/10*10
         NY = NYs - (NIM-1)/10
         do PX = 1, NPERTs
         do PY = 1, NPERTs
            do ipsf = -26, 26
            do jpsf = -26, 26
               show_psfperts((NX-1)*57*NPERTs+55*(PX-1)+28+ipsf+NPERTs,
     .                       (NY-1)*57*NPERTs+55*(PY-1)+28+jpsf+NPERTs)
     .                       = psfperts(51+ipsf,51+jpsf,PX,PY,NIM)
               enddo
               enddo
            show_psfperts((NX-1)*57*NPERTs+55*(PX-1)+28+00+NPERTs,
     .                    (NY-1)*57*NPERTs+55*(PY-1)+28+26+NPERTs)= -0.1
            show_psfperts((NX-1)*57*NPERTs+55*(PX-1)+28+00+NPERTs,
     .                    (NY-1)*57*NPERTs+55*(PY-1)+28-26+NPERTs)= -0.1
            show_psfperts((NX-1)*57*NPERTs+55*(PX-1)+28+26+NPERTs,
     .                    (NY-1)*57*NPERTs+55*(PY-1)+28+00+NPERTs)= -0.1
            show_psfperts((NX-1)*57*NPERTs+55*(PX-1)+28-26+NPERTs,
     .                    (NY-1)*57*NPERTs+55*(PY-1)+28+00+NPERTs)= -0.1
            print*,'---> NIM: ',NIM,NX,NY,PX,PY,
     .                          psfperts(51,51,PX,PY,NIM),STEM_N(NIM)
            enddo
            enddo
 
         write(NUMSTRING,'(3x,i3.3,3x)') NIM
         if (NPERTs.eq.NPERTs/2*2) then ! even number
             call img2putstr_r4(NUMSTRING,09,0.1,
     .                (NX-1)*57*NPERTs+55*((NPERTs-1)/2)+04+NPERTs,
     .                (NY-1)*57*NPERTs+55*((NPERTs-1)/2)+51+NPERTs,
     .                show_psfperts,
     .                NDIMx,NDIMy)
             call img2putstr_r4(STEM_N(NIM),09,0.1,
     .                (NX-1)*57*NPERTs+55*((NPERTs-1)/2)+04+NPERTs,
     .                (NY-1)*57*NPERTs+55*((NPERTs-1)/2)+05+NPERTs,
     .                show_psfperts,
     .                NDIMx,NDIMy)
c            print*,'-A->  ',
c    .                (NX-1)*57*NPERTs+55*((NPERTs-1)/2)+04+NPERTs,
c    .                (NY-1)*57*NPERTs+55*((NPERTs-1)/2)+51+NPERTs
c            print*,'-B->  ',
c    .                (NX-1)*57*NPERTs+55*((NPERTs-1)/2)+04+NPERTs,
c    .                (NY-1)*57*NPERTs+55*((NPERTs-1)/2)+05+NPERTs,
c    .                STEM_N(NIM)
            else
             call img2putstr_r4(NUMSTRING,09,0.1,
     .                (NX-1)*57*NPERTs+55*(NPERTs/2)+04+NPERTs,
     .                (NY-1)*57*NPERTs+55*(NPERTs/2)+51+NPERTs,
     .                show_psfperts,
     .                NDIMx,NDIMy)
             call img2putstr_r4(STEM_N(NIM),09,0.1,
     .                       (NX-1)*57*NPERTs+55*(NPERTs/2)+04+NPERTs,
     .                       (NY-1)*57*NPERTs+55*(NPERTs/2)+05+NPERTs,
     .                show_psfperts,
     .                NDIMx,NDIMy)
c            print*,'-A->  ',(NX-1)*57*NPERTs+55*(NPERTs/2)+04+NPERTs,
c    .                       (NY-1)*57*NPERTs+55*(NPERTs/2)+51+NPERTs
c            print*,'-B->  ',(NX-1)*57*NPERTs+55*(NPERTs/2)+04+NPERTs,
c    .                       (NY-1)*57*NPERTs+55*(NPERTs/2)+05+NPERTs,
c    .                       STEM_N(NIM)
             endif
 
         enddo
 
      print*,'---> OUTPUT: ',NDIMx,NDIMy
      call writfits_r4('LOG.psfperts.fits',show_psfperts,NDIMx,NDIMy)
 
      deallocate(show_psfperts)
 
 
      end
 
 
 
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/PSFs/PERT/ras2pertpsf.f"
c****                                         
c*********************************************
                                              
c--------------------------------------------------------
c
c this routine will take a set of 11x11 rasters and will
c construct a perturbed PSF out of them
c
      subroutine ras2pertpsf(pixl,psfl,il,jl,xl,yl,zl,sl,ul,Ls,
     .                       rxl,ryl,psfpertu,psfpertv,NPERTs)
      implicit none
 
      integer Ls
      real pixl(11,11,Ls)
      real psfl(11,11,Ls)
      real psfpertu(101,101,_NPRTMX_,_NPRTMX_)
      real psfpertv(101,101,_NPRTMX_,_NPRTMX_)
      integer NPERTs
      integer il(Ls), jl(Ls)
      real*8  xl(Ls), yl(Ls)
      real    sl(Ls), zl(Ls)
      real   rxl(Ls),ryl(Ls)
      integer ul(Ls)
 
      integer ipsf,jpsf
 
      integer L
      integer PX, PY
      integer ipx,ipy
 
      real ssl(99999), sslo
      real zzl(99999), zzlo
      real pixu(99999), pbar, psig
      real ptot, ftot
      integer U, Us, Uu
 
      real    el(9999)
      real    wu(99999), wt
      real    du(99999), dbar, dsig
      integer lu(9999)
 
      integer  i,  j
      integer ii, jj
      integer NIT
      integer NREJu, NREJt
 
      real    psfperto(101,101,_NPRTMX_,_NPRTMX_)
      real    psfperta(101,101,_NPRTMX_,_NPRTMX_)
      real    psfperte(101,101,_NPRTMX_,_NPRTMX_)
      integer psfpertn(101,101,_NPRTMX_,_NPRTMX_)
      integer psfpertg(101,101,_NPRTMX_,_NPRTMX_)
      real    psfpertt(101,101,_NPRTMX_,_NPRTMX_)
      real    psfperts(101,101,_NPRTMX_,_NPRTMX_)
      real    psfperth(101,101,_NPRTMX_,_NPRTMX_)
 
      real xpsf, ypsf, rpsf
      real dx, dy
      real fx, fy
      real xx, yy
      real rx, ry
      real pp, rpsf_phot, rpsfpert_phot
      real pertbar(101,101)
      real normu
      real normv
 
      integer Q, Qs
      real pq(999)
      real fq(999)
 
      real*4, dimension(:,:), allocatable ::  show_psfpertnit
 
 
      allocate(show_psfpertnit((   5    )*NPERTs*51+1-NPERTs,
     .                         (_NITPRT_)*NPERTs*51+1-NPERTs))
 
      print*,'          '
      print*,'ALLOCATE SHOW_PSFPERTNIT: ',(   5    )*NPERTs*51+1-NPERTs,
     .                              ' x ',(_NITPRT_)*NPERTs*51+1-NPERTs
      print*,'          '
 
      do i = 1,          5*NPERTs*51+1-NPERTs
      do j = 1, (_NITPRT_)*NPERTs*51+1-NPERTs
         show_psfpertnit(i,j) = -0.1
         enddo
         enddo
 
 
      if (Ls.gt.99999) stop 'ras2pertpsf: Ls.gt.99999'
 
 
c------------------------------------------------------------
c
c We start out with the central location of each
c raster in the detector and the position of the star itself,
c along with the estimate of the fraction of light in
c each pixel from the PSF; these things will not change.
c We will not assume the rasters to be sky-subtracted.
c
c------------------------------------------------------------
 
 
 
 
c------------------------------------------------------------
c
c We begin here by measuring self-consistent values
c for the flux and the sky.  The flux comes from the
c pixels within a radius of 4.5 and the sky comes
c from the PSF-model-subtracted pixels in the annulus
c between 4.5 and 6.5 pixels (restricted to being within
c the inner square of  11x11, naturally)
c
      do L = 1, Ls
         ul(L)  = 1
         zzl(L) = 0.
         ssl(L) = 0.
         do NIT = 1, 3 ! iterate to solve for flux and sky
            sslo = ssl(L)
            zzlo = zzl(L)
            Us = 0
            do i = -5, 5
            do j = -5, 5
               if (i**2+j**2.ge.4.5**2.and.i**2+j**2.le.6.5**2) then
                  Us = Us + 1
                  pixu(Us) = pixl(6+i,6+j,L)
     .                     - psfl(6+i,6+j,L)*zzl(L)
                  endif
               enddo
               enddo
            call rbarsigs(pixu,Us,pbar,psig,Uu,3.5)
            ssl(L) = pbar
            ptot = 0.
            ftot = 0.
            Qs = 0
            do i = -5, 5
            do j = -5, 5
               if (i**2+j**2.le.4.5**2) then
                  ptot = ptot + pixl(6+i,6+j,L) - ssl(L)
                  ftot = ftot + psfl(6+i,6+j,L)
                  Qs = Qs + 1
                  pq(Qs) = pixl(6+i,6+j,L) - ssl(L)
                  fq(Qs) = psfl(6+i,6+j,L)
                  endif
               enddo
               enddo
            zzl(L) = ptot/ftot
            el(L) = 0.
            do Q = 1, Qs
               el(L) = el(L) + abs(pq(Q)-zzl(L)*fq(Q))/zzl(L)
               enddo
            enddo!NIT
         do i = -5, 5
         do j = -5, 5
            ii = il(L) + i
            jj = jl(L) + j
c           write(91,191) L,ii,jj,ii-xl(L),jj-yl(L),
c    .                    pixl(6+i,6+j,L),zzl(L),ssl(L),
c    .                    psfl(6+i,6+j,L)
  191       format(i5,1x,i4.4,1x,i4.4,1x,f8.3,1x,f8.3,1x,
     .                3x,f8.1,1x,f12.4,1x,f8.2,1x,f8.6)
            enddo
            enddo
         enddo
      close(91)
 
      do L = 1, Ls
         if (el(L).gt.0.25) ul(L) = 0
         enddo
 
c------------------------------------------------------------
c
c initiate the pertpsf to 0.0
c
      do PX = 01, NPERTs
      do PY = 01, NPERTs
         do ipsf = 001, 101
         do jpsf = 001, 101
            psfpertu(ipsf,jpsf,PX,PY) = 0.00
            enddo
            enddo
         enddo
         enddo
 
 
c------------------------------------------------------------
c
c perform 5 iterations to optimize...
c
      NREJt = 0
      do NIT = 1, _NITPRT_
         NREJu = 0
         do PX = 1, NPERTs
         do PY = 1, NPERTs
            do jpsf = 001, 101
            do ipsf = 001, 101
               psfperto(ipsf,jpsf,PX,PY) = psfpertu(ipsf,jpsf,PX,PY)
               psfperta(ipsf,jpsf,PX,PY) = 0.
               psfpertn(ipsf,jpsf,PX,PY) = 0
               psfpertg(ipsf,jpsf,PX,PY) = 0
               xpsf = (ipsf-51.)/4.0
               ypsf = (jpsf-51.)/4.0
               rpsf = sqrt(xpsf**2+ypsf**2)
               if (rpsf.gt.5.0) then
                  psfpertu(ipsf,jpsf,PX,PY) = 0.
                  psfperta(ipsf,jpsf,PX,PY) = 0.
                  goto 9
                  endif
               Us = 0
               do L = 1, Ls
                  wt = max(0.,1-abs(PX-rxl(L)))*max(0.,1-abs(PY-ryl(L)))
                  if (wt.gt.0.and.ul(L).eq.1) then
                     xx = xl(L) + xpsf
                     yy = yl(L) + ypsf
                     ii = int(xx+0.5)
                     jj = int(yy+0.5)
                     i  = ii-il(L)
                     j  = jj-jl(L)
                     if (i.ge.-5.and.i.le.+5.and.
     .                   j.ge.-5.and.j.le.+5) then
                        dx = ii-xl(L)
                        dy = jj-yl(L)
                        pp = rpsfpert_phot(dx,dy,psfpertu,
     .                                     rxl(L),ryl(L),NPERTs)
                        Us = Us + 1
                        wu(Us) = wt
                        lu(Us) = L
                        du(Us) = (pixl(6+i,6+j,L)-ssl(L))/zzl(L)
     .                         -  psfl(6+i,6+j,L) - pp
                        endif
                     endif
                  enddo
               call rbarsigsw(du,wu,Us,dbar,dsig,Uu,4.0)
               do U = 1, Us
                  L = lu(u)
                  if ((.false.).and.
     .                ul(L).eq.1.and.
     .                wu(U).ge.0.5.and.
     .                abs(du(U)-dbar).gt.10*dsig) then
                      ul(L) = 0
                      NREJu = NREJu + 1
                      NREJt = NREJt + 1
                      endif
                  enddo
               psfperta(ipsf,jpsf,PX,PY) = dbar
               psfperte(ipsf,jpsf,PX,PY) = dsig
               psfpertn(ipsf,jpsf,PX,PY) = Us
               psfpertg(ipsf,jpsf,PX,PY) = Uu
    9          continue
               psfpertt(ipsf,jpsf,PX,PY) = psfperto(ipsf,jpsf,PX,PY)
     .                                   + psfperta(ipsf,jpsf,PX,PY)
               enddo
               enddo
            call smoo_psfpert(psfpertt(1,1,PX,PY),
     .                        psfperts(1,1,PX,PY))
            normu = 0.
            normv = 0.
            do ipsf = 001, 101
            do jpsf = 001, 101
               normu = normu + psfperts(ipsf,jpsf,PX,PY)/16.0
               normv = normv + psfpertv(ipsf,jpsf,PX,PY)/16.0
               enddo
               enddo
            do ipsf = 001, 101
            do jpsf = 001, 101
c
c don't do any normalization
c
c              psfperth(ipsf,jpsf,PX,PY) = psfperts(ipsf,jpsf,PX,PY)
c    .                                   - psfpertv(ipsf,jpsf,PX,PY)
c    .                                     *normu/normv
               psfperth(ipsf,jpsf,PX,PY) = psfperts(ipsf,jpsf,PX,PY)
               enddo
               enddo
            if (PX.eq.1.and.PY.eq.1) write(*,129)
            write( *,119) NIT,PX,PY,51,51,Us,Uu,
     .                              psfperto(51,51,PX,PY),
     .                              psfperta(51,51,PX,PY),
     .                              psfperte(51,51,PX,PY),
     .                              psfperts(51,51,PX,PY)-
     .                              psfperto(51,51,PX,PY),
     .                              normu,
     .                              psfperth(51,51,PX,PY)-
     .                              psfperto(51,51,PX,PY),
     .                              psfperth(51,51,PX,PY),
     .                              psfpertv(51,51,PX,PY)
  129       format(1x,'N',1x,'PX',1x,'PY',1x,' I ',1x,' J ',
     .             5x,'  NNN',1x,'  UUU',
     .             5x,'  PERT_OLD','  PERT_ADJ','  PERT_SIG',
     .                '  PERT_SMU','  PERT_CHG','  PERT_NRM',
     .                '  PERT_NEW','   PSF_VAL')
  119       format(1x,i1.1,1x,i2.2,1x,i2.2,1x,i3.3,1x,i3.3,
     .             5x,i5,1x,i5,5x,9f10.6)
            enddo!PY
            enddo!PX
         do PX = 01, NPERTs
         do PY = 01, NPERTs
            do ipsf = 001, 101
            do jpsf = 001, 101
               psfpertu(ipsf,jpsf,PX,PY) = psfperth(ipsf,jpsf,PX,PY)
               enddo
               enddo
            enddo
            enddo
         do ipsf = 001, 101
         do jpsf = 001, 101
            pertbar(ipsf,jpsf) = 0.
            do PX = 01, NPERTs
            do PY = 01, NPERTs
               pertbar(ipsf,jpsf) = pertbar(ipsf,jpsf)
     .                            + psfpertu(ipsf,jpsf,PX,PY)
     .                              /NPERTs/NPERTs
               enddo
               enddo
            enddo
            enddo
         do PX = 01, NPERTs
         do PY = 01, NPERTs
            do ipsf = 001, 101
            do jpsf = 001, 101
               if (ipsf.ge.27.and.ipsf.le.75.and.
     .             jpsf.ge.27.and.jpsf.le.75) then
                   show_psfpertnit(0*51*NPERTs+50*(PX-1)+26+(ipsf-51),
     .                       (NIT-1)*51*NPERTs+50*(PY-1)+26+(jpsf-51))
     .                               = psfperto(ipsf,jpsf,PX,PY)
                   show_psfpertnit(1*51*NPERTs+50*(PX-1)+26+(ipsf-51),
     .                       (NIT-1)*51*NPERTs+50*(PY-1)+26+(jpsf-51))
     .                               = psfperta(ipsf,jpsf,PX,PY)
                   show_psfpertnit(2*51*NPERTs+50*(PX-1)+26+(ipsf-51),
     .                       (NIT-1)*51*NPERTs+50*(PY-1)+26+(jpsf-51))
     .                               = psfpertu(ipsf,jpsf,PX,PY)
     .                               - psfperto(ipsf,jpsf,PX,PY)
                   show_psfpertnit(3*51*NPERTs+50*(PX-1)+26+(ipsf-51),
     .                       (NIT-1)*51*NPERTs+50*(PY-1)+26+(jpsf-51))
     .                               = psfpertu(ipsf,jpsf,PX,PY)
                   show_psfpertnit(4*51*NPERTs+50*(PX-1)+26+(ipsf-51),
     .                       (NIT-1)*51*NPERTs+50*(PY-1)+26+(jpsf-51))
     .                               = psfpertu(ipsf,jpsf,PX,PY)
     .                               - pertbar(ipsf,jpsf)
                   endif
               enddo
               enddo
            enddo
            enddo
         if (.false.)
     .   call writfits_r4('LOG.pertVnit.fits',show_psfpertnit,
     .                     (   5    )*NPERTs*51+1-NPERTs,
     .                     (_NITPRT_)*NPERTs*51+1-NPERTs)
 
         write(*,'(''             '')')
         write(*,'(''END PERT NIT '',2i5)') NIT,(_NITPRT_)
         write(*,'(''        Ls = '', i5)') Ls
         write(*,'(''     NREJu = '', i5)') NREJu
         write(*,'(''     NREJt = '', i5)') NREJt
         write(*,'(''             '')')
 
         enddo
 
      deallocate(show_psfpertnit)
 
      return
      end
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/PSFs/PERT/rbarsigsw.f"
c****                                         
c*********************************************
                                              
 
      subroutine rbarsigsw(xlist,wlist,NTOT,bar,sig,NUSE,SIGCLIP)
      implicit none
 
      integer NTOT
      real xlist(NTOT)
      real wlist(NTOT)
      real bar
      real sig
      integer NUSE
      real SIGCLIP
 
      integer n
      real*8  bsum, ssum, wsum
      integer nsum, nsumo
      integer NIT
 
 
      nsum = 0
      bar = 0.e0
      sig = 9e9
      do NIT = 1, 20
         nsumo = nsum
         bsum = 0.
         ssum = 0.
         wsum = 0.
         nsum = 0.
         do n = 1, NTOT
            if (abs(xlist(n)-bar).le.SIGCLIP*sig) then
               bsum = bsum + wlist(n)*xlist(n)
               ssum = ssum + wlist(n)*abs(xlist(n)-bar)
               wsum = wsum + wlist(n)
               nsum = nsum + 1
               endif
            enddo
         if (nsum.gt.0) bar = bsum/wsum
         if (nsum.gt.1) sig = ssum/wsum
         if (nsum.lt.0.35*NTOT.and.NIT.ge.3) return
         if (nsum.eq.nsumo.and.NIT.ge.3) goto 1
         enddo
  1   continue
      NUSE = nsum
      if (nsum.le.1) sig = 0.999
 
      return
      end
 
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/PSFs/PERT/rpsfpert_phot.f"
c****                                         
c*********************************************
                                              
 
      real function rpsfpert_phot(dx,dy,psfpertu,rxi,ryi,NPERTs)
      implicit none
 
      real dx, dy
      real psfpertu(101,101,_NPRTMX_,_NPRTMX_)
 
      real rxi,ryi
      integer NPERTs
      logical SHOW
 
      real    rx, ry
      integer ix, iy
      real    fx, fy
      real    pp
      real    rpsf_phot
 
      rpsfpert_phot = 0.
      if (NPERTs.eq.0) return
 
      if (NPERTs.eq.1) then
         rpsfpert_phot = rpsf_phot(dx,dy,psfpertu)
         return
         endif
 
      rx = rxi
      ry = ryi
 
      if (rx.le.   1.0) rx =    1.0
      if (ry.le.   1.0) ry =    1.0
      if (rx.gt.NPERTs) rx = NPERTs
      if (ry.gt.NPERTs) ry = NPERTs
 
      ix = min(int(rx),NPERTs-1)
      iy = min(int(ry),NPERTs-1)
      if (ix.lt.1) ix = 1
      if (iy.lt.1) iy = 1
      if (ix.gt.NPERTs-1) ix = NPERTs-1
      if (iy.gt.NPERTs-1) iy = NPERTs-1
      fx = rx-ix
      fy = ry-iy
 
      rpsfpert_phot
     .       = (1-fx)*(1-fy)*rpsf_phot(dx,dy,psfpertu(1,1,ix  ,iy  ))
     .       + ( fx )*(1-fy)*rpsf_phot(dx,dy,psfpertu(1,1,ix+1,iy  ))
     .       + (1-fx)*( fy )*rpsf_phot(dx,dy,psfpertu(1,1,ix  ,iy+1))
     .       + ( fx )*( fy )*rpsf_phot(dx,dy,psfpertu(1,1,ix+1,iy+1))
 
      return
      end
 
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/PSFs/STDPSFs/ROUTINES/infofits_stdpsf.f"
c****                                         
c*********************************************
                                              
 
c-----------------------------------------------------------------
c
c
c
 
      subroutine infofits_stdpsf(FILEI,NXPSFs,NYPSFs,ilist,jlist)
      implicit none
 
      character*80 FILEI
      integer NXPSFs
      integer NYPSFs
      integer ilist(10)
      integer jlist(10)
 
      character*2880 buffc
      byte buffb(2880)
      equivalence (buffb,buffc)
 
      integer i, k
      character*80 FILEU
 
      integer NPSFs
 
      integer q, npu, nbu
      character*08 FIELD
      character*20 STREAM
      integer ios
c
c-------------------------------------------------
c
      character*80 FILEI_stdpsf
      integer      NXPSFs_stdpsf
      integer      NYPSFs_stdpsf
      integer      ilist_stdpsf(10)
      integer      jlist_stdpsf(10)
 
      common /FILEI_stdpsf_  /FILEI_stdpsf
      common /NXPSFs_stdpsf_ /NXPSFs_stdpsf
      common /NYPSFs_stdpsf_ /NYPSFs_stdpsf
      common /ilist_stdpsf_  /ilist_stdpsf
      common /jlist_stdpsf_  /jlist_stdpsf
 
c
c-------------------------------------------------
c
 
      if (FILEI.eq.FILEI_stdpsf) then
          NXPSFs = NXPSFs_stdpsf
          NYPSFs = NYPSFs_stdpsf
          do i = 01, 10
             ilist(i) = ilist_stdpsf(i)
             jlist(i) = jlist_stdpsf(i)
             enddo
          return
          endif
 
      FILEU = 'NONE'
      do i = 75,2,-1
         if (FILEI(i:i+4).eq.'.fits') FILEU = FILEI(1:i+4)
         enddo
      if (FILEU(1:4).eq.'NONE') stop 'infofits no fits'
 
!$omp critical(infofits_stdpsf)
      open(25,file=FILEU,
     .     status='old',
     .     err   =900,
     .     iostat=ios,
     .     recl  =2880,
     .     form  ='UNFORMATTED',
     .     access='DIRECT')
 
      NXPSFs = 0
      NYPSFs = 0
      do i = 01, 10
         ilist(i) = -999
         jlist(i) = -999
         enddo
 
      read(25,rec=1) buffc
 
      do k = 01, 35
         FIELD  = buffc(01+(k-1)*80:08+(k-1)*80)
         STREAM = buffc(11+(k-1)*80:30+(k-1)*80)
         if (FIELD.eq.'NXPSFs  ') read(STREAM,*) NXPSFs
         if (FIELD.eq.'NYPSFs  ') read(STREAM,*) NYPSFs
         if (FIELD.eq.'IPSFX01 ') read(STREAM,*) ilist(01)
         if (FIELD.eq.'IPSFX02 ') read(STREAM,*) ilist(02)
         if (FIELD.eq.'IPSFX03 ') read(STREAM,*) ilist(03)
         if (FIELD.eq.'IPSFX04 ') read(STREAM,*) ilist(04)
         if (FIELD.eq.'IPSFX05 ') read(STREAM,*) ilist(05)
         if (FIELD.eq.'IPSFX06 ') read(STREAM,*) ilist(06)
         if (FIELD.eq.'IPSFX07 ') read(STREAM,*) ilist(07)
         if (FIELD.eq.'IPSFX08 ') read(STREAM,*) ilist(08)
         if (FIELD.eq.'IPSFX09 ') read(STREAM,*) ilist(09)
         if (FIELD.eq.'IPSFX10 ') read(STREAM,*) ilist(10)
         if (FIELD.eq.'JPSFY01 ') read(STREAM,*) jlist(01)
         if (FIELD.eq.'JPSFY02 ') read(STREAM,*) jlist(02)
         if (FIELD.eq.'JPSFY03 ') read(STREAM,*) jlist(03)
         if (FIELD.eq.'JPSFY04 ') read(STREAM,*) jlist(04)
         if (FIELD.eq.'JPSFY05 ') read(STREAM,*) jlist(05)
         if (FIELD.eq.'JPSFY06 ') read(STREAM,*) jlist(06)
         if (FIELD.eq.'JPSFY07 ') read(STREAM,*) jlist(07)
         if (FIELD.eq.'JPSFY08 ') read(STREAM,*) jlist(08)
         if (FIELD.eq.'JPSFY09 ') read(STREAM,*) jlist(09)
         if (FIELD.eq.'JPSFY10 ') read(STREAM,*) jlist(10)
         enddo
 
      do i = 01, NYPSFs-1
         if (jlist(i).eq.2048.and.
     .       jlist(i+1).eq.2048) jlist(i+1) = 2049
         enddo
 
      do i = 01, 10
         if (ilist(i).lt.0.or.jlist(i).lt.0) then
            print*,'infofits_stdpsf: '
            print*,'    HEADER NEEDs TO SPECIFY ALL IPSFs and JPSFs'
            print*,'       i: ',i,ilist(i),jlist(i)
            print*,'    FILE: ',FILEU
            stop
            endif
         if (NXPSFs.eq.0) stop 'infofits_stdpsf: HEADER NEEDs NXPSFs'
         if (NYPSFs.eq.0) stop 'infofits_stdpsf: HEADER NEEDs NXPSFs'
         enddo
 
      goto 901
  900 continue
      print*,' '
      print*,'ERROR ACCESSING FITS FILE IN infofits_stdpsf(): '
      write(*,'('' FILEI_stdpsf: '',a80)') FILEI_stdpsf
      write(*,'('' FILEI       : '',a80)') FILEI
      write(*,'('' FILEU       : '',a80)') FILEU
      write(*,'('' IOS         : '', i5)') IOS
      print*,' '
      stop
  901 close(25)
!$omp end critical(infofits_stdpsf)
      return
      end
 
 
c*********************************************
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/PSFs/STDPSFs/ROUTINES/loadfits_stdpsf_flexibl.f"
c****                                         
c*********************************************
                                              
 
 
c-----------------------------------------------------------------
c
c this routine will load into memory a STDPSF ; this flexible
c version will read in the median-filter PSF from a STDPBF file.
c
c
      subroutine loadfits_stdpsf(FILEI)
      implicit none
 
      character*80 FILEI
 
c
c----------------------------------------------
c
      character*80 FILEI_stdpsf
      real*4       psfij_stdpsf(101,101,10,10)
      integer      NNPSFs_stdpsf
      integer      NXPSFs_stdpsf
      integer      NYPSFs_stdpsf
      integer      ilist_stdpsf(10)
      integer      jlist_stdpsf(10)
      character*72 COMMENT1_stdpsf
      character*72 COMMENT2_stdpsf
      character*72 COMMENT3_stdpsf
      character*20 DATESTRING_stdpsf
      character*20 TIMESTRING_stdpsf
 
      common /FILEI_stdpsf_  /FILEI_stdpsf
      common /psfij_stdpsf_  /psfij_stdpsf
      common /NNPSFs_stdpsf_ /NNPSFs_stdpsf
      common /NXPSFs_stdpsf_ /NXPSFs_stdpsf
      common /NYPSFs_stdpsf_ /NYPSFs_stdpsf
      common /ilist_stdpsf_  /ilist_stdpsf
      common /jlist_stdpsf_  /jlist_stdpsf
      common /COMMENT_stdpsf_/COMMENT1_stdpsf,
     .                        COMMENT2_stdpsf,
     .                        COMMENT3_stdpsf
      common /DATESTR_stdpsf_/DATESTRING_stdpsf
      common /TIMESTR_stdpsf_/TIMESTRING_stdpsf
c
c----------------------------------------------
c
      data FILEI_stdpsf/'NONE'/
      character*80 FILEU
 
      integer NX, NY, NN
 
      integer nbyte0
      integer nbyteE
      integer nbyte1
      integer nbyte2
      integer nbper
      integer i,ios
 
      integer c1, c2, c3
 
      character*2880 buffc
      byte buffb(2880)
      equivalence (buffb,buffc)
 
      integer ii
      integer ifirst, i1, i2
      integer np1, np2, npt
 
      integer n, k
 
 
      byte b(4)
      real*4 r
      equivalence(r,b)
 
      integer q, npu, nbu
      real*4, dimension(:), allocatable :: psfn
 
      character*08 FIELD
      character*20 STREAM
 
      integer ipsf
      integer jpsf
      integer NCOMs
 
      integer NAXIS
      real    floc
      logical islinux
c
c--------------------------------------------------------
c
      if (FILEI_stdpsf.eq.FILEI) return
 
      if (FILEI(1:6).eq.'APPHOT') then
          NNPSFs_stdpsf = 1
          NXPSFs_stdpsf = 1
          NYPSFs_stdpsf = 1
          ilist_stdpsf(1) = 0000
          jlist_stdpsf(1) = 0000
          ilist_stdpsf(2) = 9999
          jlist_stdpsf(2) = 9999
          do ipsf = 001, 101
          do jpsf = 001, 101
             psfij_stdpsf(ipsf,jpsf,1,1) = 0.
             enddo
             enddo
          return
          endif
 
      FILEU = 'XXX'
      do i = 75,2,-1
         if (FILEI(i:i+4).eq.'.fits') FILEU = FILEI(1:i+4)
         enddo
      if (FILEU(1:3).eq.'XXX') then
         print*,'loadfits_stdpsf --- '
         print*,'   FILEI: ',FILEI
         print*,'   DOES NOT CONTAIN .fits'
         stop
         endif
 
      close(25)
      open(25,file=FILEU,
     .     status='old',
     .     err   =900,
     .     iostat=ios,
     .     recl  =2880,
     .     form  ='UNFORMATTED',
     .     access='DIRECT')
 
      do ii = 0001, 2880
         buffc(ii:ii) = ' '
         enddo
 
      NNPSFs_stdpsf = 0
      NXPSFs_stdpsf = 0
      NYPSFs_stdpsf = 0
 
      DATESTRING_stdpsf = 'NONE'
      TIMESTRING_stdpsf = 'NONE'
 
      do i = 01, 10
         ilist_stdpsf(i) = -999
         jlist_stdpsf(i) = -999
         enddo
 
      read(25,rec=1) buffc
      do k = 01, 35
         FIELD  = buffc(01+(k-1)*80:08+(k-1)*80)
         STREAM = buffc(11+(k-1)*80:30+(k-1)*80)
         if (FIELD.eq.'NAXIS   ') then
            read(STREAM,*) NAXIS
            if (NAXIS.eq.4) then
               close(25)
               floc = 0.
               call loadstdpsf_stdpbf(floc,FILEI,FILEI)
               return
               endif
            endif
         enddo
 
      NCOMs = 0
      COMMENT1_stdpsf = 'NO COMMENT'
      COMMENT2_stdpsf = 'NO COMMENT'
      COMMENT3_stdpsf = 'NO COMMENT'
 
      do k = 01, 35
         FIELD  = buffc(01+(k-1)*80:08+(k-1)*80)
         STREAM = buffc(11+(k-1)*80:30+(k-1)*80)
         if (FIELD.eq.'DATE    ') DATESTRING_stdpsf = STREAM
         if (FIELD.eq.'TIME    ') TIMESTRING_stdpsf = STREAM
         if (FIELD.eq.'NXPSFs  ') read(STREAM,*) NXPSFs_stdpsf
         if (FIELD.eq.'NYPSFs  ') read(STREAM,*) NYPSFs_stdpsf
         if (FIELD.eq.'IPSFX01 ') read(STREAM,*) ilist_stdpsf(01)
         if (FIELD.eq.'IPSFX02 ') read(STREAM,*) ilist_stdpsf(02)
         if (FIELD.eq.'IPSFX03 ') read(STREAM,*) ilist_stdpsf(03)
         if (FIELD.eq.'IPSFX04 ') read(STREAM,*) ilist_stdpsf(04)
         if (FIELD.eq.'IPSFX05 ') read(STREAM,*) ilist_stdpsf(05)
         if (FIELD.eq.'IPSFX06 ') read(STREAM,*) ilist_stdpsf(06)
         if (FIELD.eq.'IPSFX07 ') read(STREAM,*) ilist_stdpsf(07)
         if (FIELD.eq.'IPSFX08 ') read(STREAM,*) ilist_stdpsf(08)
         if (FIELD.eq.'IPSFX09 ') read(STREAM,*) ilist_stdpsf(09)
         if (FIELD.eq.'IPSFX10 ') read(STREAM,*) ilist_stdpsf(10)
         if (FIELD.eq.'JPSFY01 ') read(STREAM,*) jlist_stdpsf(01)
         if (FIELD.eq.'JPSFY02 ') read(STREAM,*) jlist_stdpsf(02)
         if (FIELD.eq.'JPSFY03 ') read(STREAM,*) jlist_stdpsf(03)
         if (FIELD.eq.'JPSFY04 ') read(STREAM,*) jlist_stdpsf(04)
         if (FIELD.eq.'JPSFY05 ') read(STREAM,*) jlist_stdpsf(05)
         if (FIELD.eq.'JPSFY06 ') read(STREAM,*) jlist_stdpsf(06)
         if (FIELD.eq.'JPSFY07 ') read(STREAM,*) jlist_stdpsf(07)
         if (FIELD.eq.'JPSFY08 ') read(STREAM,*) jlist_stdpsf(08)
         if (FIELD.eq.'JPSFY09 ') read(STREAM,*) jlist_stdpsf(09)
         if (FIELD.eq.'JPSFY10 ') read(STREAM,*) jlist_stdpsf(10)
         if (FIELD.eq.'COMMENT ') then
            NCOMs = NCOMs + 1
            if (NCOMs.gt.3) stop 'MORE THAN 3 COMMENTS'
            if (NCOMs.eq.1) COMMENT1_stdpsf = buffc(09+(k-1)*80:
     .                                              80+(k-1)*80)
            if (NCOMs.eq.2) COMMENT2_stdpsf = buffc(09+(k-1)*80:
     .                                              80+(k-1)*80)
            if (NCOMs.eq.3) COMMENT3_stdpsf = buffc(09+(k-1)*80:
     .                                              80+(k-1)*80)
            endif
         if (FIELD.eq.'END     ') goto 1
         enddo!k
    1 continue
 
      NNPSFs_stdpsf = NXPSFs_stdpsf*NYPSFs_stdpsf
 
      allocate(psfn(101*101*NNPSFs_stdpsf))
 
      ifirst = 2
      i = ifirst
 
      nbper  = 4*101*101*NNPSFs_stdpsf
      npt    =   101*101*NNPSFs_stdpsf
      nbyte1 = 1
      nbyte2 = nbper
      i1 = i + nbyte1/2880
      i2 = i + nbyte2/2880
 
      do i = i1, i2, 1
         nbyte0 = (i-ifirst)*2880+   1
         nbyteE = (i-ifirst)*2880+2880
         np1 = (nbyte0-nbyte1)/4 + 1
         np2 = (nbyteE-nbyte1)/4 + 1
         read(25,rec=i,iostat=ios) buffb
         do q = 001, 720
            npu = np1+q-1
            nbu = (q-1)*4
            if (.not.islinux()) stop 'not LINUX'
            b(4) = buffb(nbu+1)  ! assume LINUX ; do the byte flip
            b(3) = buffb(nbu+2)  ! this flips the bytes for r, too
            b(2) = buffb(nbu+3)
            b(1) = buffb(nbu+4)
            if (npu.le.npt) psfn(npu) = r
            enddo
         enddo
      close(25)
 
      FILEI_stdpsf  = FILEI
      do NX = 1, NXPSFs_stdpsf
      do NY = 1, NYPSFs_stdpsf
         NN = NX + (NY-1)*NXPSFs_stdpsf
         do jpsf = 001, 101
         do ipsf = 001, 101
            npu = ipsf + (jpsf-1)*101 + (NN-1)*101*101
            psfij_stdpsf(ipsf,jpsf,NX,NY) = psfn(npu)
            enddo
            enddo
         enddo
         enddo
 
      deallocate(psfn)
      return
 
  900 continue
      print*,' '
      print*,'ERROR ACCESSING FITS FILE IN loadfits_pixpsf(): '
      write(*,'('' FILEI_stdpsf: '',a80)') FILEI_stdpsf
      write(*,'('' FILEI       : '',a80)') FILEI
      write(*,'('' FILEU       : '',a80)') FILEU
      write(*,'('' IOS         : '', i5)') IOS
      print*,' '
      stop
      end
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/PSFs/STDPSFs/ROUTINES/locpsfij_stdpsf_preload.f"
c****                                         
c*********************************************
                                              
 
c-----------------------------------------------------------
c
c
c
      subroutine locpsfij_stdpsf(iloc,jloc,psfloc,FILEI)
      implicit none
 
      integer iloc
      integer jloc
      real    psfloc(101,101)
      character*(*) FILEI
c
c----------------------------------------------
c
      character*80 FILEI_stdpsf
      real*4       psfij_stdpsf(101,101,10,10)
      integer      NNPSFs_stdpsf
      integer      NXPSFs_stdpsf
      integer      NYPSFs_stdpsf
      integer      ilist_stdpsf(10)
      integer      jlist_stdpsf(10)
      character*72 COMMENT1_stdpsf
      character*72 COMMENT2_stdpsf
      character*72 COMMENT3_stdpsf
      character*20 DATESTRING_stdpsf
      character*20 TIMESTRING_stdpsf
 
      common /FILEI_stdpsf_  /FILEI_stdpsf
      common /psfij_stdpsf_  /psfij_stdpsf
      common /NNPSFs_stdpsf_ /NNPSFs_stdpsf
      common /NXPSFs_stdpsf_ /NXPSFs_stdpsf
      common /NYPSFs_stdpsf_ /NYPSFs_stdpsf
      common /ilist_stdpsf_  /ilist_stdpsf
      common /jlist_stdpsf_  /jlist_stdpsf
      common /COMMENT_stdpsf_/COMMENT1_stdpsf,
     .                        COMMENT2_stdpsf,
     .                        COMMENT3_stdpsf
      common /DATESTR_stdpsf_/DATESTRING_stdpsf
      common /TIMESTR_stdpsf_/TIMESTRING_stdpsf
c
c----------------------------------------------
c
      integer nx, ny
      real    fx, fy
 
      integer ipsf,jpsf
 
      call loadfits_stdpsf(FILEI)
 
      if (NXPSFs_stdpsf.eq.1.and.NYPSFs_stdpsf.eq.1) then
          do ipsf = 001, 101
          do jpsf = 001, 101
             psfloc(ipsf,jpsf) = psfij_stdpsf(ipsf,jpsf,1,1)
             enddo
             enddo
          return
          endif
 
      nx = 1
    1 continue
      if (iloc.gt.ilist_stdpsf(nx+1).and.nx.le.NXPSFs_stdpsf-2) then
         nx = nx + 1
         goto 1
         endif
 
      ny = 1
    2 continue
      if (jloc.gt.jlist_stdpsf(ny+1).and.ny.le.NYPSFs_stdpsf-2) then
         ny = ny + 1
         goto 2
         endif
 
      fx = 1.00*(iloc              -ilist_stdpsf(NX))/
     .          (ilist_stdpsf(NX+1)-ilist_stdpsf(NX))
      fy = 1.00*(jloc              -jlist_stdpsf(NY))/
     .          (jlist_stdpsf(NY+1)-jlist_stdpsf(NY))
 
      do ipsf = 001, 101
      do jpsf = 001, 101
         psfloc(ipsf,jpsf)
     .              = (1-fx)*(1-fy)*psfij_stdpsf(ipsf,jpsf,nx  ,ny  )
     .              + (1-fx)*( fy )*psfij_stdpsf(ipsf,jpsf,nx  ,ny+1)
     .              + ( fx )*(1-fy)*psfij_stdpsf(ipsf,jpsf,nx+1,ny  )
     .              + ( fx )*( fy )*psfij_stdpsf(ipsf,jpsf,nx+1,ny+1)
         if (psfloc(ipsf,jpsf).lt.0.00) psfloc(ipsf,jpsf) = 0.00
         enddo
         enddo
 
      return
      end
 
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/PSFs/STDPSFs/ROUTINES/stdpsf2psfbar.f"
c****                                         
c*********************************************
                                              
 
c----------------------------------------------------
c
c this routine will take a standard PSF and will find
c the average ; could do a better job area-averaging
c to deal with the outer edges and gap edges.
c
      subroutine stdpsf2psfbar(PSFFILE_USE,psfbar)
      implicit none
 
      character*80 PSFFILE_USE
      real*4  psfbar(101,101)
 
      integer NXPSFs, NYPSFs
      integer ilist(10)
      integer jlist(10)
      real*4  psfloc(101,101)
      integer ipsf, jpsf
      integer i, j
 
      call infofits_stdpsf(PSFFILE_USE,NXPSFs,NYPSFs,ilist,jlist)
 
      do i = 001, 101
      do j = 001, 101
         psfbar(i,j) = 0.
         enddo
         enddo
 
      do i = 1, NXPSFs
      do j = 1, NYPSFs
         call locpsfij_stdpsf(ilist(i),jlist(j),
     .                        psfloc,PSFFILE_USE)
         do ipsf = 001, 101
         do jpsf = 001, 101
            psfbar(ipsf,jpsf) = psfbar(ipsf,jpsf) + psfloc(ipsf,jpsf)
            enddo
            enddo
         enddo
         enddo
 
      do ipsf = 001, 101
      do jpsf = 001, 101
         psfbar(ipsf,jpsf) = psfbar(ipsf,jpsf)/NXPSFs/NYPSFs
         enddo
         enddo
 
      return
      end
 
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/PSFs/STDPSFs/ROUTINES/writfits_stdpsf.f"
c****                                         
c*********************************************
                                              
 
c-----------------------------------------------------------------
c
c
c
 
      subroutine writfits_stdpsf(FILE,pix,NXPSFs,NYPSFs,ilist,jlist,
     .                           COMMENT1,COMMENT2,COMMENT3)
      implicit none
 
      character*80 FILE
      integer NXPSFs
      integer NYPSFs
      real*4  pix(*)
      integer ilist(10)
      integer jlist(10)
      character*(*) COMMENT1
      character*(*) COMMENT2
      character*(*) COMMENT3
 
      character*72 COMMENT1U
      character*72 COMMENT2U
      character*72 COMMENT3U
 
      integer nbyte0
      integer nbyteE
      integer nbyte1
      integer nbyte2
      integer nbper
      integer i,ios
 
      integer c1, c2, c3
 
      character*2880 buffc
      byte buffb(2880)
      equivalence (buffb,buffc)
 
      integer ii
      integer ifirst, i1, i2
      integer np1, np2, npt
 
      integer n, k
      character*80 FILEU
 
      character*20 DATESTRING
      character*20 TIMESTRING
      integer today(3)
      integer now(3)
 
      integer NPSFs
 
      byte b(4)
      real*4 r
      equivalence(r,b)
 
      integer q, npu, nbu
 
 
      FILEU = FILE
      do i = 75,2,-1
         if (FILE(i:i+4).eq.'.fits') FILEU = FILE(1:i+4)
         enddo
 
      open(10,file=FILEU,
     .     status='unknown',
     .     err   =900,
     .     recl  =2880,
     .     form  ='UNFORMATTED',
     .     access='DIRECT')
 
      do ii = 0001, 2880
         buffc(ii:ii) = ' '
         enddo
 
      COMMENT1U = COMMENT1
      COMMENT2U = COMMENT2
      COMMENT3U = COMMENT3
 
      do i = 72, 02, -1
         if (COMMENT1U(i:i).eq.'!') COMMENT1U = COMMENT1U(1:i-1)
         if (COMMENT2U(i:i).eq.'!') COMMENT2U = COMMENT2U(1:i-1)
         if (COMMENT3U(i:i).eq.'!') COMMENT3U = COMMENT3U(1:i-1)
         enddo
 
 
c
c we'll only use ONE header record, for simplicity ; it still
c conforms to the FITS standard, but I know to read only one
c
      call idate(today)
      call itime(now)
 
c 12 1 2013
c 10 24 17
 
      write(DATESTRING,'(a1,i4.4,''-'',i2.2,''-'',i2.2,a1)')
     .                   '''',today(3), today(2), today(1),''''
      write(TIMESTRING,'(a1,i2.2,'':'',i2.2,'':'',i2.2,a1)')
     .                   '''',now(1), now(2), now(3),''''
 
      NPSFs = NXPSFs*NYPSFs
 
      write(buffc( 0*80+1: 1*80),'(''SIMPLE  =                    T'')')
      write(buffc( 1*80+1: 2*80),'(''BITPIX  =                  -32'')')
      write(buffc( 2*80+1: 3*80),'(''NAXIS   = '',i20)') 3
      write(buffc( 3*80+1: 4*80),'(''NAXIS1  = '',i20)') 101
      write(buffc( 4*80+1: 5*80),'(''NAXIS2  = '',i20)') 101
      write(buffc( 5*80+1: 6*80),'(''NAXIS3  = '',i20)') NPSFs
      write(buffc( 6*80+1: 7*80),'(''DATE    = '',a20)') DATESTRING
      write(buffc( 7*80+1: 8*80),'(''TIME    = '',a20)') TIMESTRING
      write(buffc( 8*80+1:09*80),'(''BSCALE  = '',f20.4)') 00001.00
      write(buffc( 9*80+1:10*80),'(''BZERO   = '',f20.4)') 00000.00
      write(buffc(10*80+1:11*80),'(''NXPSFs  = '',i20)') NXPSFs
      write(buffc(11*80+1:12*80),'(''NYPSFs  = '',i20)') NYPSFs
      do n = 01, 10
         k = 11+n
         write(buffc(k*80+1:k*80+80),'(''IPSFX'',i2.2,'' = '',i20)')
     .                                          n,ilist(n)
         enddo
      do n = 01, 10
         k = 21+n
         write(buffc(k*80+1:k*80+80),'(''JPSFY'',i2.2,'' = '',i20)')
     .                                          n,jlist(n)
         enddo
      write(buffc(32*80+1:34*80),'(''COMMENT '',a72)') COMMENT1U
      write(buffc(33*80+1:34*80),'(''COMMENT '',a72)') COMMENT2U
      write(buffc(34*80+1:35*80),'(''COMMENT '',a72)') COMMENT3U
      write(buffc(35*80+1:36*80),'(''END                           '')')
      write(10,rec=1,iostat=ios) buffc
 
      ifirst = 2
      i = ifirst
 
      nbper  = 4*101*101*NPSFs
      npt    =   101*101*NPSFs
      nbyte1 = 1
      nbyte2 = nbper
      i1 = i + nbyte1/2880
      i2 = i + nbyte2/2880
 
      do i = i1, i2, 1
         nbyte0 = (i-ifirst)*2880+   1
         nbyteE = (i-ifirst)*2880+2880
         np1 = (nbyte0-nbyte1)/4 + 1
         np2 = (nbyteE-nbyte1)/4 + 1
         do q = 001, 720
            npu = np1+q-1
            nbu = (q-1)*4
            if (npu.ge.1.and.npu.le.npt) r = pix(npu)
            if (.not.(_LINUX_)) then
               buffb(nbu+1) = b(1)
               buffb(nbu+2) = b(2)
               buffb(nbu+3) = b(3)
               buffb(nbu+4) = b(4)
               endif
            if ((_LINUX_)) then
               buffb(nbu+1) = b(4)
               buffb(nbu+2) = b(3)
               buffb(nbu+3) = b(2)
               buffb(nbu+4) = b(1)
               endif
            enddo
         write(10,rec=i,iostat=ios) buffc
         enddo
 
      close(10)
 
      return
 
  900 continue
      print*,' '
      print*,'ERROR ACCESSING FITS FILE IN writfits_pixpsf(): '
      write(*,'('' FILE : '',a80)') FILE
      write(*,'('' FILEU: '',a80)') FILEU
      print*,' '
      stop
      end
 
 
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/PSFs/STDPSFs/ROUTINES/savefits_stdpsf.f"
c****                                         
c*********************************************
                                              
 
c-----------------------------------------------------------------
c
c
c
      subroutine savefits_stdpsf(FILEO)
      implicit none
 
      character*(*) FILEO
 
c     integer NXPSFs
c     integer NYPSFs
c     integer ilist(10)
c     integer jlist(10)
 
      integer nbyte0
      integer nbyteE
      integer nbyte1
      integer nbyte2
      integer nbper
      integer i,ios
 
      integer c1, c2, c3
 
      character*2880 buffc
      byte buffb(2880)
      equivalence (buffb,buffc)
 
      integer ii
      integer ifirst, i1, i2
      integer np1, np2, npt
 
      integer n, k
      character*80 FILEU
 
      integer today(3)
      integer now(3)
 
      byte b(4)
      real*4 r
      equivalence(r,b)
 
      integer q, npu, nbu
      integer NX, NY, NN
      integer ipsf, jpsf
 
c
c----------------------------------------------
c
      character*80 FILEI_stdpsf
      integer      NNPSFs_stdpsf
      integer      NXPSFs_stdpsf
      integer      NYPSFs_stdpsf
      integer      ilist_stdpsf(10)
      integer      jlist_stdpsf(10)
      character*72 COMMENT1_stdpsf
      character*72 COMMENT2_stdpsf
      character*72 COMMENT3_stdpsf
      character*20 DATESTRING_stdpsf
      character*20 TIMESTRING_stdpsf
      real*4       psfij_stdpsf(101,101,10,10)
 
      common /FILEI_stdpsf_  /FILEI_stdpsf
      common /NNPSFs_stdpsf_ /NNPSFs_stdpsf
      common /NXPSFs_stdpsf_ /NXPSFs_stdpsf
      common /NYPSFs_stdpsf_ /NYPSFs_stdpsf
      common /ilist_stdpsf_  /ilist_stdpsf
      common /jlist_stdpsf_  /jlist_stdpsf
      common /COMMENT_stdpsf_/COMMENT1_stdpsf,
     .                        COMMENT2_stdpsf,
     .                        COMMENT3_stdpsf
      common /DATESTR_stdpsf_/DATESTRING_stdpsf
      common /TIMESTR_stdpsf_/TIMESTRING_stdpsf
      common /psfij_stdpsf_  /psfij_stdpsf
 
      logical islinux
c
c----------------------------------------------
c
 
      FILEI_stdpsf = FILEO
 
      FILEU = FILEO
      do i = 75,2,-1
         if (FILEO(i:i+4).eq.'.fits') FILEU = FILEO(1:i+4)
         enddo
 
      open(10,file=FILEU,
     .     status='unknown',
     .     err   =900,
     .     recl  =2880,
     .     form  ='UNFORMATTED',
     .     access='DIRECT')
 
      do ii = 0001, 2880
         buffc(ii:ii) = ' '
         enddo
 
c
c we'll only use ONE header record, for simplicity ; it still
c conforms to the FITS standard, but I know to read only one
c
      call idate(today)
      call itime(now)
 
      write(DATESTRING_stdpsf,'(a1,i4.4,''-'',i2.2,''-'',i2.2,a1)')
     .                       '''',today(3), today(2), today(1),''''
      write(TIMESTRING_stdpsf,'(a1,i2.2,'':'',i2.2,'':'',i2.2,a1)')
     .                       '''',now(1),   now(2),   now(3),  ''''
 
      NNPSFs_stdpsf = NXPSFs_stdpsf*NYPSFs_stdpsf
 
      write(buffc( 0*80+1: 1*80),'(''SIMPLE  =                    T'')')
      write(buffc( 1*80+1: 2*80),'(''BITPIX  =                  -32'')')
      write(buffc( 2*80+1: 3*80),'(''NAXIS   = '',i20)') 3
      write(buffc( 3*80+1: 4*80),'(''NAXIS1  = '',i20)') 101
      write(buffc( 4*80+1: 5*80),'(''NAXIS2  = '',i20)') 101
      write(buffc( 5*80+1: 6*80),'(''NAXIS3  = '',i20)') NNPSFs_stdpsf
      write(buffc( 6*80+1: 7*80),'(''DATE    = '',a20)')
     .                               DATESTRING_stdpsf
      write(buffc( 7*80+1: 8*80),'(''TIME    = '',a20)')
     .                               TIMESTRING_stdpsf
      write(buffc( 8*80+1:09*80),'(''BSCALE  = '',f20.4)') 00001.00
      write(buffc( 9*80+1:10*80),'(''BZERO   = '',f20.4)') 00000.00
      write(buffc(10*80+1:11*80),'(''NXPSFs  = '',i20)') NXPSFs_stdpsf
      write(buffc(11*80+1:12*80),'(''NYPSFs  = '',i20)') NYPSFs_stdpsf
      do n = 01, 10
         k = 11+n
         write(buffc(k*80+1:k*80+80),'(''IPSFX'',i2.2,'' = '',i20)')
     .                                          n,ilist_stdpsf(n)
         enddo
      do n = 01, 10
         k = 21+n
         write(buffc(k*80+1:k*80+80),'(''JPSFY'',i2.2,'' = '',i20)')
     .                                          n,jlist_stdpsf(n)
         enddo
      write(buffc(32*80+1:34*80),'(''COMMENT '',a72)') COMMENT1_stdpsf
      write(buffc(33*80+1:34*80),'(''COMMENT '',a72)') COMMENT2_stdpsf
      write(buffc(34*80+1:35*80),'(''COMMENT '',a72)') COMMENT3_stdpsf
      write(buffc(35*80+1:36*80),'(''END                           '')')
      write(10,rec=1,iostat=ios) buffc
 
      ifirst = 2
      i = ifirst
 
      nbper  = 4*101*101*NNPSFs_stdpsf
      npt    =   101*101*NNPSFs_stdpsf
      nbyte1 = 1
      nbyte2 = nbper
      i1 = i + nbyte1/2880
      i2 = i + nbyte2/2880
 
      do i = i1, i2, 1
         nbyte0 = (i-ifirst)*2880+   1
         nbyteE = (i-ifirst)*2880+2880
         np1 = (nbyte0-nbyte1)/4 + 1
         np2 = (nbyteE-nbyte1)/4 + 1
         do q = 001, 720
            npu = np1+q-1
            nbu = (q-1)*4
            NN   = 1 + (npu-1)/101/101
            NY   = 1 + (NN-1)/NXPSFs_stdpsf
            NX   = NN - (NY-1)*NXPSFs_stdpsf
            jpsf = 1 +  (npu-1-(NN-1)*101*101)/101
c           ipsf = npu -     jpsf*101 - (NN-1)*101*101    ! old
            ipsf = npu - (jpsf-1)*101 - (NN-1)*101*101    ! fixed
            if (npu.ge.1.and.npu.le.npt)
     .          r = psfij_stdpsf(ipsf,jpsf,NX,NY)
            if (.not.islinux()) stop 'must be LINUX now...'
            buffb(nbu+1) = b(4)
            buffb(nbu+2) = b(3)
            buffb(nbu+3) = b(2)
            buffb(nbu+4) = b(1)
            enddo
         write(10,rec=i,iostat=ios) buffc
         enddo
 
      close(10)
 
      return
 
  900 continue
      print*,' '
      print*,'ERROR ACCESSING FITS FILE IN savefits_stdpsf(): '
      write(*,'('' FILEO: '',a80)') FILEO
      write(*,'('' FILEU: '',a80)') FILEU
      print*,' '
      stop
      end
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/PSFs/STDPSFs/ROUTINES/writfits_psfpert2std.f"
c****                                         
c*********************************************
                                              
 
      subroutine writfits_psfpert2std(psfpert,NPERTs,STEM9,
     .                                NAXIS1,NAXIS2,
     .                                PSFFILE_LIB,
     .                                PSFFILE_USE)
      implicit none
 
      real         psfpert(101,101,_NPRTMX_,_NPRTMX_)
      integer      NPERTs
      character*09 STEM9
      integer      NAXIS1
      integer      NAXIS2
      character*80 PSFFILE_LIB
      character*80 PSFFILE_USE
 
      real*4, dimension(:,:,:), allocatable :: psfijn
 
      integer PX, PY
      integer ipsf,jpsf
 
      integer NPSFs,  NPSF
      integer NXPSFs, NXPSF
      integer NYPSFs, NYPSF
      integer ilist(10)
      integer jlist(10)
      real*4  psfloc(101,101)
      integer i, j, ii
      real    xpsf, ypsf
      real    rx, ry
      real    rpsfpert_phot
 
      character*72 COMMENT1
      character*72 COMMENT2
      character*72 COMMENT3
 
      real atot
      real ctot
 
 
      print*,'ENTER psfpert2std...'
      print*,'      NPERTs: ',NPERTs
      print*,' PSFFILE_LIB: ',PSFFILE_LIB
      print*,' PSFFILE_USE: ',PSFFILE_USE
 
      call infofits_stdpsf(PSFFILE_LIB,NXPSFs,NYPSFs,ilist,jlist)
 
      print*,'      NXPSFs: ',NXPSFs
      print*,'      NYPSFs: ',NYPSFs
      NPSFs = NXPSFs*NYPSFs
      print*,'       NPSFs: ',NPSFs
 
      allocate(psfijn(101,101,NPSFs))
 
      atot = 0.
      ctot = 0.
      do NXPSF = 1, NPERTs
      do NYPSF = 1, NPERTs
         ctot = ctot + psfpert(051,051,NXPSF,NYPSF)
         do ipsf = 001, 101
         do jpsf = 001, 101
            atot = atot + abs(psfpert(ipsf,jpsf,NXPSF,NYPSF)/
     .                                NPERTs/NPERTs/16)
            enddo
            enddo
         enddo
         enddo
 
      do NXPSF = 1, NXPSFs
      do NYPSF = 1, NYPSFs
         NPSF = 1 + (NXPSF-1) + (NYPSF-1)*NXPSFs
         do ipsf = 001, 101
         do jpsf = 001, 101
            psfijn(ipsf,jpsf,NPSF) = 0.0
            enddo
            enddo
         enddo
         enddo
 
      NPSF = 0
      do NXPSF = 1, NXPSFs
      do NYPSF = 1, NYPSFs
         NPSF = 1 + (NXPSF-1) + (NYPSF-1)*NXPSFs
         call locpsfij_stdpsf(ilist(NXPSF),jlist(NYPSF),
     .                        psfijn(1,1,NPSF),PSFFILE_LIB)
 
         rx = 1. + 1.*ilist(NXPSF)*(NPERTs-1)/NAXIS1
         ry = 1. + 1.*jlist(NYPSF)*(NPERTs-1)/NAXIS2
         do ipsf = 001, 101
         do jpsf = 001, 101
            xpsf = (ipsf-51)*0.25
            ypsf = (jpsf-51)*0.25
            psfloc(ipsf,jpsf) = rpsfpert_phot(xpsf,ypsf,psfpert,
     .                                        rx,ry,NPERTs)
            psfijn(ipsf,jpsf,NPSF) = psfijn(ipsf,jpsf,NPSF)
     .                             + psfloc(ipsf,jpsf)
            enddo
            enddo
         write(*,199) NXPSF,NYPSF,NPSF,
     .                ilist(NXPSF),jlist(NYPSF),rx,ry,
     .                psfijn(51,51,NPSF),psfloc(51,51),
     .                psfijn(51,51,NPSF)-psfloc(51,51)
  199    format(1x,i2.2,1x,i2.2,1x,i3.3,1x,i4.4,1x,i4.4,1x,
     .          3x,f8.3,1x,f8.3,5x,f10.5,5x,f10.5,5x,f10.5)
         enddo
         enddo
 
      ii = 80
      do i = 80, 56, -1
         if (PSFFILE_LIB(i:i).ne.' ') ii = i
         enddo
 
      COMMENT1 = 'PSFFILE_LIB: ' // PSFFILE_LIB(ii-55:ii)
      write(COMMENT2,198) STEM9,NPERTs
      write(COMMENT3,197) atot, ctot
  198 format('FOR STEM=',a9,' AND NPERTs = ',i1)
  197 format('AVG ABS PERT: ',f8.5,10x,'AVG CEN PERT: ',f8.5)
 
      call writfits_stdpsf(PSFFILE_USE,psfijn,
     .                     NXPSFs,NYPSFs,ilist,jlist,
     .                     COMMENT1,COMMENT2,COMMENT3)
 
      print*,' '
      do j = NYPSFs, 1, -1
         write(*,137) j,(psfijn(51,51,(j-1)*NXPSFs+i),i=1,NXPSFs)
  137    format(10x,i3,1x,10f10.5)
         enddo
 
      deallocate(psfijn)
 
      end
 
 
 
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/PSFs/STDPBFs/ROUTINES/infofits_stdpbf.f"
c****                                         
c*********************************************
                                              
 
c-----------------------------------------------------------------
c
c
c
 
      subroutine infofits_stdpbf(FILE,NXPSFs,NYPSFs,Fs,ilist,jlist)
      implicit none
 
      character*80 FILE
      integer NXPSFs
      integer NYPSFs
      integer Fs
      integer ilist(10)
      integer jlist(10)
 
      character*2880 buffc
      byte buffb(2880)
      equivalence (buffb,buffc)
 
      integer i, k
      character*80 FILEU
 
      integer NPSFs
 
      integer q, npu, nbu
      character*08 FIELD
      character*20 STREAM
 
      if (FILE(1:1).eq.'0') then
         NXPSFs = 2
         NYPSfs = 2
         ilist(1) = 0000
         ilist(2) = 9999
         jlist(1) = 0000
         jlist(2) = 9999
         Fs = 1
         return
         endif
 
      FILEU = 'NONE'
      do i = 75,2,-1
         if (FILE(i:i+4).eq.'.fits') FILEU = FILE(1:i+4)
         enddo
 
!$omp critical(infofits_stdpsf)
      print*,'OPEN25 FILE  = ',FILE
      print*,'OPEN25 FILEU = ',FILEU
      close(25)
      open(25,file=FILEU,
     .     status='old',
     .     err   =900,
     .     recl  =2880,
     .     access='DIRECT')
 
      NXPSFs = 0
      NYPSFs = 0
      do i = 01, 10
         ilist(i) = -999
         jlist(i) = -999
         enddo
 
      read(25,rec=1) buffc
 
      Fs = 0
      NXPSFs = 0
      NYPSFs = 0
      do k = 01, 35
         FIELD  = buffc(01+(k-1)*80:08+(k-1)*80)
         STREAM = buffc(11+(k-1)*80:30+(k-1)*80)
         if (FIELD.eq.'NXPSFs  ') read(STREAM,*) NXPSFs
         if (FIELD.eq.'NYPSFs  ') read(STREAM,*) NYPSFs
         if (FIELD.eq.'NAXIS4')   read(STREAM,*) Fs
         if (FIELD.eq.'IPSFX01 ') read(STREAM,*) ilist(01)
         if (FIELD.eq.'IPSFX02 ') read(STREAM,*) ilist(02)
         if (FIELD.eq.'IPSFX03 ') read(STREAM,*) ilist(03)
         if (FIELD.eq.'IPSFX04 ') read(STREAM,*) ilist(04)
         if (FIELD.eq.'IPSFX05 ') read(STREAM,*) ilist(05)
         if (FIELD.eq.'IPSFX06 ') read(STREAM,*) ilist(06)
         if (FIELD.eq.'IPSFX07 ') read(STREAM,*) ilist(07)
         if (FIELD.eq.'IPSFX08 ') read(STREAM,*) ilist(08)
         if (FIELD.eq.'IPSFX09 ') read(STREAM,*) ilist(09)
         if (FIELD.eq.'IPSFX10 ') read(STREAM,*) ilist(10)
         if (FIELD.eq.'JPSFY01 ') read(STREAM,*) jlist(01)
         if (FIELD.eq.'JPSFY02 ') read(STREAM,*) jlist(02)
         if (FIELD.eq.'JPSFY03 ') read(STREAM,*) jlist(03)
         if (FIELD.eq.'JPSFY04 ') read(STREAM,*) jlist(04)
         if (FIELD.eq.'JPSFY05 ') read(STREAM,*) jlist(05)
         if (FIELD.eq.'JPSFY06 ') read(STREAM,*) jlist(06)
         if (FIELD.eq.'JPSFY07 ') read(STREAM,*) jlist(07)
         if (FIELD.eq.'JPSFY08 ') read(STREAM,*) jlist(08)
         if (FIELD.eq.'JPSFY09 ') read(STREAM,*) jlist(09)
         if (FIELD.eq.'JPSFY10 ') read(STREAM,*) jlist(10)
         enddo
 
 
      do i = 01, NYPSFs-1
         if (jlist(i  ).eq.2048.and.
     .       jlist(i+1).eq.2048) jlist(i+1) = 2049
         enddo
 
 
 
      do i = 01, 10
         if (ilist(i).lt.0.or.jlist(i).lt.0) then
            print*,'infofits_stdpsf: '
            print*,'    HEADER NEEDs TO SPECIFY ALL IPSFs and JPSFs'
            print*,'       i: ',i,ilist(i),jlist(i)
            print*,'    FILE: ',FILEU
            stop
            endif
         if (NXPSFs.eq.0) stop 'infofits_stdpsf: HEADER NEEDs NXPSFs'
         if (NYPSFs.eq.0) stop 'infofits_stdpsf: HEADER NEEDs NXPSFs'
         enddo
 
      goto 901
 
 
  900 continue
      print*,' '
      print*,'ERROR ACCESSING FITS FILE IN infofits_stdpsf(): '
      write(*,'('' FILE : '',a80)') FILE
      write(*,'('' FILEU: '',a80)') FILEU
      print*,' '
      stop
  901 close(25)
!$omp end critical(infofits_stdpsf)
      return
      end
 
 
c*********************************************
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/PSFs/STDPBFs/ROUTINES/locpsfijf_stdpbf.f"
c****                                         
c*********************************************
                                              
 
      subroutine locpsfijf_stdpbf(iloc,jloc,floc,psfloc,FILEI)
      implicit none
 
      integer iloc
      integer jloc
      real    floc ! this is the focus level
      real    psfloc(101,101)
 
      character*80 FILEI
 
c
c-----------------------------------------
c
 
      integer ipsf, jpsf
 
      integer F
      real    rf
      real    ff, fx, fy
 
 
      character*80 FILEI_stdpbf
      integer      NX, NXPSFs_stdpbf
      integer      NY, NYPSFs_stdpbf
      integer      NN, NNPSFs_stdpbf    ! total num of spatial PSFs
      integer      NF, NFPSFs_stdpbf    ! number of focus levels
      integer      ilist_stdpbf(10)
      integer      jlist_stdpbf(10)
 
      common /FILEI_stdpbf_ /FILEI_stdpbf
      common /NXPSFs_stdpbf_/NXPSFs_stdpbf
      common /NYPSFs_stdpbf_/NYPSFs_stdpbf
      common /NNPSFs_stdpbf_/NNPSFs_stdpbf
      common /NFPSFs_stdpbf_/NFPSFs_stdpbf
      common /ilist_stdpbf_ /ilist_stdpbf
      common /jlist_stdpbf_ /jlist_stdpbf
 
      real psfijxyf_stdpbf(101,101,10,10,11) ! maximum of 10x10 zones and 11 focus levels
      common /psfijxyf_stdpbf_/psfijxyf_stdpbf
 
c
c-----------------------------------------
c
 
      if (FILEI(1:1).eq.'0') then
          do ipsf = 001, 101
          do jpsf = 001, 101
             psfloc(ipsf,jpsf) = 0.
             enddo
             enddo
         return
         endif
 
      if (FILEI.ne.FILEI_stdpbf) call loadfits_stdpbf(FILEI)
 
      nx = 1
    1 continue
      if (iloc.gt.ilist_stdpbf(nx+1).and.nx.le.NXPSFs_stdpbf-2) then
         nx = nx + 1
         goto 1
         endif
      fx = 1.00*(iloc              -ilist_stdpbf(NX))/
     .          (ilist_stdpbf(NX+1)-ilist_stdpbf(NX))
 
      ny = 1
    2 continue
      if (jloc.gt.jlist_stdpbf(ny+1).and.ny.le.NYPSFs_stdpbf-2) then
         ny = ny + 1
         goto 2
         endif
      fy = 1.00*(jloc              -jlist_stdpbf(NY))/
     .          (jlist_stdpbf(NY+1)-jlist_stdpbf(NY))
 
      rf = floc
      if (rf.le.0.99) then
         rf = (1+NFPSFs_stdpbf)/2.0
         endif
      if (rf.lt.1) rf = 1.0
      if (rf.gt.NFPSFs_stdpbf-0.001) rf = NFPSFs_stdpbf-0.0001
      nf = int(rf)
      ff = rf-nf
 
      do ipsf = 001, 101
      do jpsf = 001, 101
        psfloc(ipsf,jpsf)
     .  = (1-fx)*(1-fy)*(1-ff)*psfijxyf_stdpbf(ipsf,jpsf,nx  ,ny  ,nf  )
     .  + (1-fx)*( fy )*(1-ff)*psfijxyf_stdpbf(ipsf,jpsf,nx  ,ny+1,nf  )
     .  + ( fx )*(1-fy)*(1-ff)*psfijxyf_stdpbf(ipsf,jpsf,nx+1,ny  ,nf  )
     .  + ( fx )*( fy )*(1-ff)*psfijxyf_stdpbf(ipsf,jpsf,nx+1,ny+1,nf  )
     .  + (1-fx)*(1-fy)*( ff )*psfijxyf_stdpbf(ipsf,jpsf,nx  ,ny  ,nf+1)
     .  + (1-fx)*( fy )*( ff )*psfijxyf_stdpbf(ipsf,jpsf,nx  ,ny+1,nf+1)
     .  + ( fx )*(1-fy)*( ff )*psfijxyf_stdpbf(ipsf,jpsf,nx+1,ny  ,nf+1)
     .  + ( fx )*( fy )*( ff )*psfijxyf_stdpbf(ipsf,jpsf,nx+1,ny+1,nf+1)
        if (psfloc(ipsf,jpsf).lt.0.00) psfloc(ipsf,jpsf) = 0.00
        enddo
        enddo
 
      return
      end
 
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/PSFs/STDPBFs/ROUTINES/findfocus_stdpbf.f"
c****                                         
c*********************************************
                                              
 
c------------------------------------------------------------
c
c this routine will determine the optimal focus for this
c image
c
c
      subroutine findfocus_stdpbf(pixc,HMB,HIFLAG,NAXIS1,NAXIS2,
     .                            FOCUS_LEVEL,Ns,PSFFILE_USE,VERBOSE)
      implicit none
 
      integer      NAXIS1, NAXIS2
      real         pixc(NAXIS1,NAXIS2)
      real         HMB
      integer      HIFLAG
      real         FOCUS_LEVEL
      integer      Ns
      character*80 PSFFILE_USE
      logical      VERBOSE
 
      integer i, ii, iii
      integer j, jj, jjj
 
      real*8 xr, yr, fr, sr
      real   qr
      real   qmin
 
      real*8 xro, yro
 
      real   dx, dy
 
      real    FEST
      integer hobs, fnd_hloc
      integer pobs, fnd_ploc
      real    mbar_sky
      integer L, Ls, Lu
      integer il(99999)
      integer jl(99999)
 
      real*8 , dimension(:), allocatable :: xl
      real*8 , dimension(:), allocatable :: yl
      real*4 , dimension(:), allocatable :: ml
      real*4 , dimension(:), allocatable :: zl
      real*4 , dimension(:), allocatable :: sl
      real*4 , dimension(:), allocatable :: ql
      real*4 , dimension(:), allocatable :: pl
      integer, dimension(:), allocatable :: ul
      real*4 , dimension(:,:,:), allocatable :: pixl
      real*4 , dimension(:,:,:), allocatable :: psfl
      real*4   qbar, qsig
      real*4   psfq
 
      real    rpsf_phot
      real    psfloc(101,101)
      integer NSs, NSu
 
      real    fx, fy
 
      real   FMIN
      integer NIT
 
      integer FF
      real    rF, qtot
      common /qmin_/qmin
 
c
c-----------------------------------------------------------
c
 
      FMIN = HIFLAG
 
      if (VERBOSE) then
          print*,'                                              '
          print*,'ENTER FINDFOCUS:                              '
          print*,'                                              '
          print*,'   FIND THE AVERAGE DIFFERENCE BETWEEN THE    '
          print*,'   INPUT LIBRARY PSF AND THE PSF IN THIS IMAGE'
          print*,'                                              '
          print*,'         HIFLAG: ',HIFLAG
          print*,'           FMIN: ',FMIN
          print*,'            HMB: ',HMB
          print*,'            NIT: ',NIT
          print*,'         NAXIS1: ',NAXIS1
          print*,'         NAXIS2: ',NAXIS2
          print*,'            PSF: ',PSFFILE_USE
          print*,'      FOCUS_LEV: ',FOCUS_LEVEL
          print*,'                '
          endif
 
      Ls  = 0
      NIT = 0
  3   continue
      NIT = NIT + 1
      if (VERBOSE)
     .print*,'FIND_PSFFOCUS FIND STARS TO USE --- NIT: ',NIT,FMIN,Ls
 
      Ls = 0
      do jj = 10, NAXIS2-10
      do ii = 10, NAXIS1-10
         if (pixc(ii,jj).lt.FMIN/9) goto 222
         if (pixc(ii,jj).gt.HIFLAG) goto 222
         FEST = pixc(ii-1,jj+1) + pixc(ii,jj+1) + pixc(ii+1,jj+1)
     .        + pixc(ii-1,jj  ) + pixc(ii,jj  ) + pixc(ii+1,jj  )
     .        + pixc(ii-1,jj-1) + pixc(ii,jj-1) + pixc(ii+1,jj-1)
     .        - HMB*9
         if (FEST.lt. FMIN) goto 222
         if (pixc(ii,jj)-HMB.gt.0.666*FEST) goto 222 ! likely CR
         do i = -9, 9
         do j = -9, 9
            if (i**2+j**2.le.9.5**2) then
               if (pixc(ii+i,jj+j).gt.pixc(ii,jj)) goto 222
               endif
            if (i**2+j**2.le.5.5**2) then
               if (pixc(ii+i,jj+j).lt.-50) goto 222
               endif
            enddo
            enddo
 
         Ls = Ls + 1              ! being included
         if (Ls.gt.99999) then
            print*,'  Ls    : ', Ls
            print*,'       ii: ', ii
            print*,'       jj: ', jj
            print*,'  99999: ',99999
            stop
            endif
         il(Ls) = ii
         jl(Ls) = jj
 222     continue
         enddo
         enddo
 
      if (Ls.lt.10) then
         print*,' '
         print*,'NEED TO HAVE AT LEAST 10 FOCUS-DONOR STARS'
         print*,'IN THE EXPOSURE'
         print*,' '
         FMIN = FMIN/2.0
         if (FMIN.lt.2500) then
            print*,'FMIN: ',FMIN
            print*,'  Ls: ',Ls
            Ns = Ls
            FOCUS_LEVEL = -1.00
            return
            endif
         goto 3
         endif
 
      if (VERBOSE) print*,'FIND_PSFPERT: allocate Ls: ',Ls
 
      allocate(xl(Ls))
      allocate(yl(Ls))
      allocate(ml(Ls))
      allocate(zl(Ls))
      allocate(sl(Ls))
      allocate(ql(Ls))
      allocate(pl(Ls))
      allocate(ul(Ls))
      allocate(pixl(11,11,Ls))
      allocate(psfl(11,11,Ls))
 
 
      Lu = 0
      do L = 1, Ls
         ii = il(L)
         jj = jl(L)
         fx = (pixc(ii+1,jj)-pixc(ii-1,jj))/2/
     .        (pixc(ii  ,jj)-min(pixc(ii+1,jj),pixc(ii-1,jj)))
         fy = (pixc(ii,jj+1)-pixc(ii,jj-1))/2/
     .        (pixc(ii,jj  )-min(pixc(ii,jj+1),pixc(ii,jj-1)))
         xr = ii + fx
         yr = jj + fy
         xro = xr
         yro = yr
         sr = HMB
         call locpsfijf_stdpbf(ii,jj,0.0,psfloc,PSFFILE_USE) ! use the average PSF for now...
         call find_xyzXX_NAXIS(xr,yr,fr,sr,qr,pixc,psfloc,
     .                         NAXIS1,NAXIS2,0) ! 0 means weight by flux (center focus)
         if (qr.gt.0.50) goto 333  ! this may help prevent CRs from
         Lu = Lu + 1
         il(Lu) = ii
         jl(Lu) = jj
         xl(Lu) = xr
         yl(Lu) = yr
         ml(Lu) = -2.5*log10(max(fr,1.))
         zl(Lu) = fr
         sl(Lu) = sr
         ql(Lu) = qr
         pl(Lu) = pixc(ii,jj)
         ul(Lu) = 1
         do i = 01, 11
         do j = 01, 11
            iii = ii+(i-6)
            jjj = jj+(j-6)
            dx  = iii-xl(Lu)
            dy  = jjj-yl(Lu)
            pixl(i,j,Lu) = pixc(iii,jjj) - sl(Lu)
            psfl(i,j,Lu) = rpsf_phot(dx,dy,psfloc)
            enddo
            enddo
         if (((Lu.lt.00010).or.
     .        (Lu.lt.00100.and.L.eq.L/0010*0010).or.
     .        (Lu.lt.01000.and.L.eq.L/0025*0025).or.
     .        (L.eq.Ls)).and.VERBOSE)
     .   write( *,182) Lu,il(Lu),jl(Lu),
     .                 xl(Lu),yl(Lu),ml(Lu),zl(Lu),sl(Lu),
     .                 psfl(6,6,Lu),ql(Lu),
     .                 pixc(ii,jj),psfloc(51,51)
 182     format(i5.5,1x,i4.4,1x,i4.4,
     .          3x,f9.3,1x,f9.3,1x,f8.4,1x,f10.1,1x,f7.2,
     .          3x,f8.6,1x,f8.6,4x,f9.1,1x,f8.4)
 333     continue
         enddo
 
      qmin = 999999.9
      fmin = 0.
      do FF = 010, 110, 02
         rF = FF/10.0
         qtot = 0.
         do L = 1, Lu
            sr = HMB
            ii = il(L)
            jj = jl(L)
            xr = xl(L)
            yr = yl(L)
            call locpsfijf_stdpbf(ii,jj,rF,psfloc,PSFFILE_USE)
            call find_xyzXX_NAXIS(xr,yr,fr,sr,qr,pixc,psfloc,
     .                            NAXIS1,NAXIS2,0) ! 0 means weight by flux (center focus)
            qtot = qtot + qr/Lu
            enddo
         if (VERBOSE) write( *,133) rF, qtot
  133    format(1x,f8.3,1x,f9.4)
         if (qtot.lt.qmin) then
             qmin = qtot
             fmin = rF
             endif
         enddo
 
      Ns = Ls
      FOCUS_LEVEL = fmin
      if (VERBOSE) then
         print*,' '
         print*,' '
         print*,'-----> FOCUS_LEVEL = ',FOCUS_LEVEL,qmin
         print*,' '
         print*,' '
         endif
 
      return
      end
 
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/PSFs/STDPBFs/ROUTINES/loadstdpsf_stdpbf.f"
c****                                         
c*********************************************
                                              
 
c----------------------------------------------------
c
c This routine will take floc (a value for the
c focus-level) and a PBF file and will interpolate
c betwen the focus levels to generate a standard psf,
c which can be used by the program or saved somewhere.
c If the focus level is given to be less than 1,
c then the routine will use the median focus level.
c
      subroutine loadstdpsf_stdpbf(floc,FILEI,FILEF)
      implicit none
 
      real    floc       ! this is the focus level
      character*80 FILEI ! ths is the STDPBF file
      character*80 FILEF
 
      integer ipsf, jpsf
 
      integer F
      real    rf
      real    ff, fx, fy
 
      integer i
      integer NX, NY, NN, NF
c
c-----------------------------------------------------------
c
      character*80 FILEI_stdpbf
      integer      NXPSFs_stdpbf
      integer      NYPSFs_stdpbf
      integer      NNPSFs_stdpbf    ! total num of spatial PSFs
      integer      NFPSFs_stdpbf    ! number of focus levels
      integer      ilist_stdpbf(10)
      integer      jlist_stdpbf(10)
      character*72 COMMENT1_stdpbf
      character*72 COMMENT2_stdpbf
      character*20 DATESTRING_stdpbf
      character*20 TIMESTRING_stdpbf
 
      common /FILEI_stdpbf_ /FILEI_stdpbf
      common /NXPSFs_stdpbf_/NXPSFs_stdpbf
      common /NYPSFs_stdpbf_/NYPSFs_stdpbf
      common /NNPSFs_stdpbf_/NNPSFs_stdpbf
      common /NFPSFs_stdpbf_/NFPSFs_stdpbf
      common /ilist_stdpbf_ /ilist_stdpbf
      common /jlist_stdpbf_ /jlist_stdpbf
      common /COMMENT_stdpbf_/COMMENT1_stdpbf,
     .                        COMMENT2_stdpbf
      common /DATESTR_stdpbf_/DATESTRING_stdpbf
      common /TIMESTR_stdpbf_/TIMESTRING_stdpbf
 
      real psfijxyf_stdpbf(101,101,10,10,11) ! maximum of 10x10 zones and 11 focus levels
      common /psfijxyf_stdpbf_/psfijxyf_stdpbf
c
c-----------------------------------------------------------
c
      character*80 FILEI_stdpsf
      integer      NNPSFs_stdpsf
      integer      NXPSFs_stdpsf
      integer      NYPSFs_stdpsf
      integer      ilist_stdpsf(10)
      integer      jlist_stdpsf(10)
      character*72 COMMENT1_stdpsf
      character*72 COMMENT2_stdpsf
      character*72 COMMENT3_stdpsf
      character*20 DATESTRING_stdpsf
      character*20 TIMESTRING_stdpsf
      real*4       psfij_stdpsf(101,101,10,10)
 
      common /FILEI_stdpsf_  /FILEI_stdpsf
      common /NNPSFs_stdpsf_ /NNPSFs_stdpsf
      common /NXPSFs_stdpsf_ /NXPSFs_stdpsf
      common /NYPSFs_stdpsf_ /NYPSFs_stdpsf
      common /ilist_stdpsf_  /ilist_stdpsf
      common /jlist_stdpsf_  /jlist_stdpsf
      common /COMMENT_stdpsf_/COMMENT1_stdpsf,
     .                        COMMENT2_stdpsf,
     .                        COMMENT3_stdpsf
      common /DATESTR_stdpsf_/DATESTRING_stdpsf
      common /TIMESTR_stdpsf_/TIMESTRING_stdpsf
      common /psfij_stdpsf_  /psfij_stdpsf
c
c----------------------------------------------
c
 
 
c
c make sure the FILEU PBF is currently loaded
c into the common block
c
      print*,'ENTER loadstdpsf_stdpbf...'
 
      print*,' CALL loadfits_stdpbf...'
      call loadfits_stdpbf(FILEI)
 
 
 
c
c determine which of the STDPBF focus-related PSFs to
c interpolate between in order to make the single-focus
c STDPSF
c
      print*,' DETERMINE floc = ',floc
      rf = floc
      if (rf.le.0.99) then
         rf = (1+NFPSFs_stdpbf)/2.0
         floc = rf
         endif
      if (rf.lt.1) rf = 1.0
      if (rf.gt.NFPSFs_stdpbf-0.001) rf = NFPSFs_stdpbf-0.0001
      nf = int(rf)
      ff = rf-nf
 
      print*,'       USE floc = ',floc
 
      do NX = 1, 10
      do NY = 1, 10
         do ipsf = 001, 101
         do jpsf = 001, 101
            psfij_stdpsf(ipsf,jpsf,NX,NY)
     .          = (1-ff)*psfijxyf_stdpbf(ipsf,jpsf,NX,NY,NF  )
     .          + ( ff )*psfijxyf_stdpbf(ipsf,jpsf,NX,NY,NF+1)
            enddo
            enddo
         enddo
         enddo
 
 
c
c copy over the fiducial spatial info from the
c 5-d stdpbf to the 4-d stdpsf
c
      do i = 01, 10
         ilist_stdpsf(i) = ilist_stdpbf(i)
         jlist_stdpsf(i) = jlist_stdpbf(i)
         enddo
 
      FILEI_stdpsf  = FILEF
 
      NXPSFs_stdpsf = NXPSFs_stdpbf
      NYPSFs_stdpsf = NYPSFs_stdpbf
      NNPSFs_stdpsf = NNPSFs_stdpbf
 
      DATESTRING_stdpsf = DATESTRING_stdpbf
      TIMESTRING_stdpsf = TIMESTRING_stdpbf
 
      COMMENT1_stdpsf = COMMENT1_stdpbf
      COMMENT2_stdpsf = COMMENT2_stdpbf
      write(COMMENT3_stdpsf,'(''FOCUS LEVEL = '',f8.3)') floc
 
 
      return
      end
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/PSFs/STDPBFs/ROUTINES/loadfits_stdpbf.f"
c****                                         
c*********************************************
                                              
 
 
c----------------------------------------------------------------------
c
c This routine will load in a STDPBF file into the common
c block that will allow it to be used by whatever program
c is calling.
c
c
      subroutine loadfits_stdpbf(FILEI)
      implicit none
 
      character*80 FILEI
 
      integer i
      integer ipsf, jpsf
      integer NX, NY, NN, NF
      character*80 FILEU
 
      character*2880 buffc
      byte buffb(2880)
      equivalence (buffb,buffc)
 
      integer k
 
      character*08 FIELD
      character*20 STREAM
 
      real, dimension(:,:,:,:), allocatable :: psf_ijnf
 
      integer BITPIX
      integer NAXIS1, NAXIS2, NAXIS3, NAXIS4
      real    BZERO
      real    BSCALE
 
      integer NCOMs
c
c-----------------------------------------
c
      character*80 FILEI_stdpbf
      data FILEI_stdpbf/'NONE'/
 
      integer NXPSFs_stdpbf
      integer NYPSFs_stdpbf
      integer NNPSFs_stdpbf   ! total num of spatial PSFs
      integer NFPSFs_stdpbf   ! number of focus levels
 
      integer ilist_stdpbf(10)
      integer jlist_stdpbf(10)
 
      common /FILEI_stdpbf_ /FILEI_stdpbf
      common /NXPSFs_stdpbf_/NXPSFs_stdpbf
      common /NYPSFs_stdpbf_/NYPSFs_stdpbf
      common /NNPSFs_stdpbf_/NNPSFs_stdpbf
      common /NFPSFs_stdpbf_/NFPSFs_stdpbf
      common /ilist_stdpbf_ /ilist_stdpbf
      common /jlist_stdpbf_ /jlist_stdpbf
 
      character*72 COMMENT1_stdpbf
      character*72 COMMENT2_stdpbf
      common /COMMENT_stdpbf_/COMMENT1_stdpbf,
     .                        COMMENT2_stdpbf
      character*20 DATESTRING_stdpsf
      character*20 TIMESTRING_stdpsf
      common /DATESTRING_stdpbf_/DATESTRING_stdpsf
      common /TIMESTRING_stdpbf_/TIMESTRING_stdpsf
 
      real psfijxyf_stdpbf(101,101,10,10,11) ! maximum of 11 focus levels and 10 zones
      common /psfijxyf_stdpbf_/psfijxyf_stdpbf
 
      integer ios
c
c-----------------------------------------
c
 
      if (FILEI_stdpbf.eq.FILEI) return
 
 
      FILEU = 'NONE'
      do i = 75,2,-1
         if (FILEI(i:i+4).eq.'.fits') FILEU = FILEI(1:i+4)
         enddo
 
      close(25)
      open(25,file=FILEU,
     .     status='old',
     .     err   =900,
     .     recl  =2880,
     .     iostat=ios,
     .     access='DIRECT')
 
      FILEI_stdpbf = FILEI
 
      NXPSFs_stdpbf = 0
      NYPSFs_stdpbf = 0
      do i = 01, 10
         ilist_stdpbf(i) = -999
         jlist_stdpbf(i) = -999
         enddo
      COMMENT1_stdpbf = 'NO COMMENT'
      COMMENT2_stdpbf = 'NO COMMENT'
 
      read(25,rec=1) buffc
      close(25) ! this routine only needs to read the header
 
      BITPIX = 0
      NAXIS1 = 0
      NAXIS2 = 0
      NAXIS3 = 0
      NAXIS4 = 0
      BZERO  = -1
      BSCALE = 0
 
      NCOMs = 0
      do k = 01, 35
         FIELD  = buffc(01+(k-1)*80:08+(k-1)*80)
         STREAM = buffc(11+(k-1)*80:30+(k-1)*80)
         if (FIELD.eq.'BITPIX  ') read(STREAM,*) BITPIX
         if (FIELD.eq.'NAXIS1')   read(STREAM,*) NAXIS1
         if (FIELD.eq.'NAXIS2')   read(STREAM,*) NAXIS2
         if (FIELD.eq.'NAXIS3')   read(STREAM,*) NAXIS3
         if (FIELD.eq.'NAXIS4')   read(STREAM,*) NAXIS4
         if (FIELD.eq.'BZERO ')   read(STREAM,*) BZERO
         if (FIELD.eq.'BSCALE')   read(STREAM,*) BSCALE
         if (FIELD.eq.'NXPSFs  ') read(STREAM,*) NXPSFs_stdpbf
         if (FIELD.eq.'NYPSFs  ') read(STREAM,*) NYPSFs_stdpbf
         if (FIELD.eq.'NAXIS4')   read(STREAM,*) NFPSFs_stdpbf
         if (FIELD.eq.'IPSFX01 ') read(STREAM,*) ilist_stdpbf(01)
         if (FIELD.eq.'IPSFX02 ') read(STREAM,*) ilist_stdpbf(02)
         if (FIELD.eq.'IPSFX03 ') read(STREAM,*) ilist_stdpbf(03)
         if (FIELD.eq.'IPSFX04 ') read(STREAM,*) ilist_stdpbf(04)
         if (FIELD.eq.'IPSFX05 ') read(STREAM,*) ilist_stdpbf(05)
         if (FIELD.eq.'IPSFX06 ') read(STREAM,*) ilist_stdpbf(06)
         if (FIELD.eq.'IPSFX07 ') read(STREAM,*) ilist_stdpbf(07)
         if (FIELD.eq.'IPSFX08 ') read(STREAM,*) ilist_stdpbf(08)
         if (FIELD.eq.'IPSFX09 ') read(STREAM,*) ilist_stdpbf(09)
         if (FIELD.eq.'IPSFX10 ') read(STREAM,*) ilist_stdpbf(10)
         if (FIELD.eq.'JPSFY01 ') read(STREAM,*) jlist_stdpbf(01)
         if (FIELD.eq.'JPSFY02 ') read(STREAM,*) jlist_stdpbf(02)
         if (FIELD.eq.'JPSFY03 ') read(STREAM,*) jlist_stdpbf(03)
         if (FIELD.eq.'JPSFY04 ') read(STREAM,*) jlist_stdpbf(04)
         if (FIELD.eq.'JPSFY05 ') read(STREAM,*) jlist_stdpbf(05)
         if (FIELD.eq.'JPSFY06 ') read(STREAM,*) jlist_stdpbf(06)
         if (FIELD.eq.'JPSFY07 ') read(STREAM,*) jlist_stdpbf(07)
         if (FIELD.eq.'JPSFY08 ') read(STREAM,*) jlist_stdpbf(08)
         if (FIELD.eq.'JPSFY09 ') read(STREAM,*) jlist_stdpbf(09)
         if (FIELD.eq.'JPSFY10 ') read(STREAM,*) jlist_stdpbf(10)
         if (FIELD.eq.'COMMENT ') then
            NCOMs = NCOMs + 1
            if (NCOMs.gt.2) stop '.gt.2 COMMENTS IN AN STDPBF FILE'
            if (NCOMs.eq.1) COMMENT1_stdpbf = buffc(09+(k-1)*80:
     .                                              80+(k-1)*80)
            if (NCOMs.eq.2) COMMENT2_stdpbf = buffc(09+(k-1)*80:
     .                                              80+(k-1)*80)
            endif
         if (FIELD.eq.'END     ') goto 3
         enddo
    3 continue
 
      if (BZERO.ne.0.or.
     .    BSCALE.ne.1.or.
     .    NAXIS1.ne.101.or.
     .    NAXIS2.ne.101.or.
     .    NAXIS3.gt.100.or.
     .    NAXIS4.gt. 11) then
          print*,' '
          print*,'STDPBF FITS FILE NOT IN COMPLIANCE: '
          print*,'  FILEU: ',FILEU
          print*,'                  '
          print*,'FIGURE OUT WHERE: '
          print*,'  BSCALE must be ==  1:  BSCALE: ',BSCALE
          print*,'  BZERO  must be ==  0:  BZERO : ',BZERO
          print*,'  NAXIS1 must be ==101:  NAXIS1: ',NAXIS1
          print*,'  NAXIS2 must be ==101:  NAXIS2: ',NAXIS2
          print*,'  NAXIS3 must be <=100:  NAXIS3: ',NAXIS3
          print*,'  NAXIS4 must be <= 11:  NAXIS4: ',NAXIS4
          print*,'                  '
          print*,'                  '
          do k = 01, 35
             FIELD  = buffc(01+(k-1)*80:08+(k-1)*80)
             STREAM = buffc(11+(k-1)*80:30+(k-1)*80)
             write(*,'(i2.2,1x,a80)') k,buffc(01+(k-1)*80:80+(k-1)*80)
             enddo
          print*,'                  '
          stop   'in STDPBF'
          endif
 
      do i = 01, 10
         if (ilist_stdpbf(i).lt.0.or.jlist_stdpbf(i).lt.0) then
            print*,'                 '
            print*,'infofits_stdpsf: '
            print*,'                 '
            print*,'  HEADER NEEDs TO SPECIFY ALL IPSFs and JPSFs'
            print*,'                 i : ',i
            print*,'    ilist_stdpbf(i): ',ilist_stdpbf(i)
            print*,'    jlist_stdpbf(i): ',jlist_stdpbf(i)
            print*,'               FILE: ',FILEU
            print*,'                     '
            stop
            endif
         enddo
 
      do i = 01, NYPSFs_stdpbf-1
         if (jlist_stdpbf(i  ).eq.2048.and.
     .       jlist_stdpbf(i+1).eq.2048) jlist_stdpbf(i+1) = 2049
         enddo
 
      write(*,'(''#                          '')')
      write(*,'(''#  LOADFITS_STDPBF '')')
      write(*,'(''#                          '')')
      write(*,'(''#   FILEI_stdpbf: '',80a)') FILEI_stdpbf
      write(*,'(''#  NXPSFs_stdpbf: '',i4.2,1x,10i6)')
     .               NXPSFs_stdpbf,(ilist_stdpbf(NX),NX=1,NXPSFs_stdpbf)
      write(*,'(''#  NYPSFs_stdpbf: '',i4.2,1x,10i6)')
     .               NYPSFs_stdpbf,(jlist_stdpbf(NY),NY=1,NYPSFs_stdpbf)
      NNPSFs_stdpbf = NXPSFs_stdpbf*NYPSFs_stdpbf
      write(*,'(''#  NNPSFs: '',i4.2)') NNPSFs_stdpbf
      write(*,'(''#  NFPSFs: '',i4.2)') NFPSFs_stdpbf
      write(*,'(''#                          '')')
 
      allocate(psf_ijnf(101,101,NNPSFs_stdpbf,NFPSFs_stdpbf))
      call readfits_r4_4D(FILEU,psf_ijnf,101,101,NNPSFs_stdpbf,
     .                                           NFPSFs_stdpbf)
 
      do NX = 1, 10
      do NY = 1, 10
      do NF = 1, 11
         do ipsf = 001, 101
         do jpsf = 001, 101
            psfijxyf_stdpbf(ipsf,jpsf,NX,NY,NF) = 0.
            enddo
            enddo
         enddo
         enddo
         enddo
 
      do NF = 01, NFPSFs_stdpbf
         do NX = 1, NXPSFs_stdpbf
         do NY = 1, NYPSFs_stdpbf
            NN = NX + (NY-1)*NXPSFs_stdpbf
            do ipsf = 001, 101
            do jpsf = 001, 101
               psfijxyf_stdpbf(ipsf,jpsf,NX,NY,NF) = 0.
               enddo
               enddo
            do ipsf = 001, 101
            do jpsf = 001, 101
               i = psf_ijnf(ipsf,jpsf,NN,NF)
               enddo
               enddo
            do ipsf = 001, 101
            do jpsf = 001, 101
               psfijxyf_stdpbf(ipsf,jpsf,NX,NY,NF) =
     .                psf_ijnf(ipsf,jpsf,NN,NF)
               enddo
               enddo
            enddo
            enddo
         enddo
 
      deallocate(psf_ijnf)
 
      return
 
  900 continue
      print*,' '
      print*,'ERROR ACCESSING FITS FILE IN loadfits_stdpbf(): '
      print*,' '
      write(*,'('' FILEI_stdpbf: '',a80)') FILEI_stdpbf
      write(*,'('' FILEI       : '',a80)') FILEI
      write(*,'('' FILEU       : '',a80)') FILEU
      print*,' '
      write(*,'(''       iostat: '',i5)')  ios
      print*,' '
      stop
 
 
      end
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/PSFs/STDPSFs/ROUTINES/rpsf_phot_ij_stdpsf.f"
c****                                         
c*********************************************
                                              
 
 
c--------------------------------------------------------
c
c this routine evaluates the PSF at a pixel that is at a
c particular location relative to the center of a star (dx,dy)
c for a star that it at a particular location in an image (iloc,jloc).
c The PSF in PSFFILE is evaluated.  It uses the standard
c PSF format and only reads in a new PSF if the filename
c has changed.
c
      real function rpsf_phot_ij_STDPSF(dx,dy,iloc,jloc,PSFFILE)
      implicit none
 
      real          dx         ! input x offset of pixel from star center
      real          dy         ! input y offset of pixel from star center
      integer       iloc,jloc  ! input integer location of star central pixel
      character*(*) PSFFILE    ! STDPSF file name
 
c
c--------------------------------------------------------------
c
 
      integer nx, ny
      real    fx, fy
      real    rpsf_phot
      integer i,j
 
c
c-------------------------------------------------
c
c These are the variables from the STDPSF common
c block that are needed to evaluate the PSF
c
      real*4       psfij_stdpsf(101,101,10,10)
      integer      NNPSFs_stdpsf
      integer      NXPSFs_stdpsf
      integer      NYPSFs_stdpsf
      integer      ilist_stdpsf(10)
      integer      jlist_stdpsf(10)
 
      common /psfij_stdpsf_  /psfij_stdpsf
      common /NNPSFs_stdpsf_ /NNPSFs_stdpsf
      common /NXPSFs_stdpsf_ /NXPSFs_stdpsf
      common /NYPSFs_stdpsf_ /NYPSFs_stdpsf
      common /ilist_stdpsf_  /ilist_stdpsf
      common /jlist_stdpsf_  /jlist_stdpsf
c
c-------------------------------------------------
c
 
c
c this routine will read in a new STDPSF into
c the current buffer; if PSFFILE is already read
c in, it will do nothing
c
      call loadfits_stdpsf(PSFFILE)
 
 
c-----------------------------------------------------------------
c
c determine which two x-columns we must interpolate between
c for this particular location in the image
c
      nx = 1
    1 continue
      if (iloc.gt.ilist_stdpsf(nx+1).and.nx.le.NXPSFs_stdpsf-2) then
         nx = nx + 1
         goto 1
         endif
 
c-----------------------------------------------------------------
c
c determine which two y-rows we must interpolate between
c for this particular location in the image
c
      ny = 1
    2 continue
      if (jloc.gt.jlist_stdpsf(ny+1).and.ny.le.NYPSFs_stdpsf-2) then
         ny = ny + 1
         goto 2
         endif
 
c-----------------------------------------------------------------
c
c determine where we are between the columns and rows
c
      fx = 1.00*(iloc              -ilist_stdpsf(NX))/
     .          (ilist_stdpsf(NX+1)-ilist_stdpsf(NX))
      fy = 1.00*(jloc              -jlist_stdpsf(NY))/
     .          (jlist_stdpsf(NY+1)-jlist_stdpsf(NY))
 
      if (NXPSFs_stdpsf.eq.1) then
         fx = 0.0
         NX = 1
         endif
      if (NYPSFs_stdpsf.eq.1) then
         fy = 0.0
         NY = 1
         endif
 
c-----------------------------------------------------------------
c
c now, evaluate the PSF at the four local fiducial locations, then
c do a simple bi-linear interpolation
c
      rpsf_phot_ij_STDPSF
     .    = (1-fx)*(1-fy)*rpsf_phot(dx,dy,psfij_stdpsf(1,1,NX  ,NY  ))
     .    + (1-fx)*( fy )*rpsf_phot(dx,dy,psfij_stdpsf(1,1,NX  ,NY+1))
     .    + ( fx )*(1-fy)*rpsf_phot(dx,dy,psfij_stdpsf(1,1,NX+1,NY  ))
     .    + ( fx )*( fy )*rpsf_phot(dx,dy,psfij_stdpsf(1,1,NX+1,NY+1))
 
      return
      end
 
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/ROUTINES/IMAGE/pix2mask.f"
c****                                         
c*********************************************
                                              
 
c------------------------------------------------------------------
c
c there are three parameters for this model:
c
c    0) distance mapping --- dgen(NRAD)
c    1) radial   profile --- pgen(C,NRAD)
c    2) spike    profile --- sgen(C,NRAD)
c    3) spike    width   --- wgen(C,NRAD)
c
c
      subroutine pix2mask(pixm,KKK,
     .                    BBs,iref_bb,jref_bb,zref_bb,
     .                    NAXIS1,NAXIS2)
      implicit none
 
      integer NAXIS1
      integer NAXIS2
      real    pixm(NAXIS1,NAXIS2)
      integer KKK
 
      integer BBs
      integer iref_bb(BBs)
      integer jref_bb(BBs)
      real    zref_bb(BBs)
c
c----------------------------------------------------------
c
      integer BB
 
      integer dgen(23)
      common /dgen_/dgen
 
      real rgen(10,23) ! profile for radial part
      common /rgen_/rgen
 
      real sgen(10,23) ! profile for spike
      common /sgen_/sgen
 
      integer wgen(10,23) ! spike width
      common /wgen_/wgen
 
      real ANGLE_XC(4,10) ! x angle at the 4 corners   !  3  4
      common /angle_xc_/angle_xc                       !  1  2
      data angle_xc/
     .  000.00, 000.00, 000.00, 000.00, !  1
     .  000.00, 000.00, 000.00, 000.00, !  2
     .  000.00, 000.00, 000.00, 000.00, !  3
     .  000.00, 000.00, 000.00, 000.00, !  4
     .   -1.40,  -1.30,  -4.00,  -2.30, !  5
     .  000.00, 000.00, 000.00, 000.00, !  6
     .   -6.00,  -6.00,  -6.00,  -6.00, !  7
     .   43.02,  43.29,  43.03,  43.11, !  8
     .  048.10, 048.10, 048.10, 048.10, !  9
     .  000.00, 000.00, 000.00, 000.00/ ! 10
 
      real ANGLE_YC(4,10) ! y angle at the 4 corners
      common /angle_yc_/angle_yc
      data angle_yc/
     .  090.00, 090.00, 090.00, 090.00, !  1
     .  090.00, 090.00, 090.00, 090.00, !  2
     .  090.00, 090.00, 090.00, 090.00, !  3
     .  090.00, 090.00, 090.00, 090.00, !  4
     .  092.40, 091.50, 093.80, 091.60, !  5
     .  090.00, 090.00, 090.00, 090.00, !  6
     .  090.00, 090.00, 090.00, 090.00, !  7
     .  -46.21, -46.78, -46.84, -47.39, !  8
     .  -48.10, -48.10, -48.10, -48.10, !  9
     .  090.00, 090.00, 090.00, 090.00/ ! 10
 
      integer ANGLE_EC(10) ! extent ... where are the corners?
      common /angle_ec_/angle_ec
      data angle_ec/
     .   800,  800,  800,  800, 4096,
     .  4096, 1024, 4096, 4096, 1014/
 
      integer F
      real fx, fy, fr
      real angle_x, angle_y
      real cosx, sinx
      real cosy, siny
      real mu, zu, zr
      integer i, iu, ic, ii
      integer j, ju, jc, jj
      integer NR
      integer ir, di, q
      real    rr, rru
      real    fxl(4), fyl(4)
      character*80 FILENAME
      real    rprofi(1001)
      real    sprofi(1001)
      real    wprofi(1001)
      real    dxa, dya
      real    dxb, dyb
      integer IRs
      real*8  xcor, ycor
      real*8  xraw, yraw
      integer iraw, jraw
      real*8  xcor0,ycor0
 
      real mask0(2001,2001)
      real mask1(2001,2001)
      real mask2(2001,2001)
      real mask3(2001,2001)
      real xx, yy
      real    z, zmax
      integer FMAX
      integer ia1(2001), ib1(2001), ja1, jb1
      integer ia2(2001), ib2(2001), ja2, jb2
      integer ia3(2001), ib3(2001), ja3, jb3
 
      real    rprof_ik
      real    sprof_ik
      integer wprof_ik
      integer ib, jb
      real    fxold, fyold
      integer ffold
      logical same
      real    xr_uv, yr_uv
 
      common /fxyf/ fxold, fyold, ffold
 
      data fxold / -1.0/
      data fyold / -1.0/
      data ffold / -1.0/
 
      real pixm_sum(4096,4096) ! the sum of them
      real pixm_max(4096,4096) ! the max of them
 
      fxold = fx
      fyold = fy
      ffold = F
 
      if (.true.) then
         print*,'                 '
         print*,'ENTER MASK_TILE: '
         print*,'            KKK: ',KKK
         print*,'            BBs: ',BBs
         print*,'                 '
         endif
 
      if (NAXIS1.ne.4096.or.
     .    NAXIS2.ne.4096) stop 'mask NAXIS hardcode probs'
 
      do i = 001, 4096
      do j = 001, 4096
         pixm_max(i,j) = 0.
         pixm_sum(i,j) = 0.
         enddo
         enddo
 
      do BB = 1, BBs
         if (BB.eq.BB/10*10) print*,'---> BB: ',BB,' / ',BBs
 
         xraw = iref_bb(BB)
         yraw = jref_bb(BB)
         zu   = zref_bb(BB)
 
         fx = 0.5
         fy = 0.5
         if (KKK.eq.5) then ! ACS/WFC
            fx = xraw/4096.0
            fy = yraw/4096.0
            endif
         if (KKK.eq.7) then ! ACS/HRC
            fx = xraw/1024.0
            fy = yraw/1024.0
            endif
         if (KKK.eq.8) then ! WFC3/UVIS
            fx = xraw/4096.0
            fy = yraw/4096.0
            endif
         if (KKK.eq.8) then ! WFC3/UVIS
            fx = xraw/4096.0
            fy = yraw/4096.0
            endif
         if (fx.lt.0.00) fx = 0.0
         if (fy.lt.0.00) fy = 0.0
         if (fx.gt.1.00) fx = 1.0
         if (fy.gt.1.00) fy = 1.0
 
         ANGLE_X = (1-fx)*(1-fy)*ANGLE_XC(1,KKK)
     .           + ( fx )*(1-fy)*ANGLE_XC(2,KKK)
     .        + (1-fx)*( fy )*ANGLE_XC(3,KKK)
     .        + ( fx )*( fy )*ANGLE_XC(4,KKK)
         ANGLE_Y = (1-fx)*(1-fy)*ANGLE_YC(1,KKK)
     .           + ( fx )*(1-fy)*ANGLE_YC(2,KKK)
     .           + (1-fx)*( fy )*ANGLE_YC(3,KKK)
     .           + ( fx )*( fy )*ANGLE_YC(4,KKK)
         cosx = cos(ANGLE_X*3.14159/180)
         sinx = sin(ANGLE_X*3.14159/180)
         cosy = cos(ANGLE_Y*3.14159/180)
         siny = sin(ANGLE_Y*3.14159/180)
         ja1 = 2001
         ja2 = 2001
         ja3 = 2001
         jb1 = 0001
         jb2 = 0001
         jb3 = 0001
         do j = 0001, 2001
            ia1(j) = 2001
            ib1(j) = 0001
            ia2(j) = 2001
            ib2(j) = 0001
            ia3(j) = 2001
            ib3(j) = 0001
            do i = 0001, 2001
               mask0(i,j) = 0.00
               mask1(i,j) = 0.00
               mask2(i,j) = 0.00
               mask3(i,j) = 0.00
               ir = 1 + sqrt((i-1001.)**2+(j-1001.)**2)
               if (ir.le.1001) then
                  mask1(i,j) = mask1(i,j) + rprof_ik(ir,KKK)
                  dya = (i-1001)*sinx - (j-1001)*cosx
                  if (abs(dya).lt.wprof_ik(ir,KKK)) then
                     mask2(i,j) = mask2(i,j) + sprof_ik(ir,KKK)
                     endif
                  dyb = (i-1001)*siny - (j-1001)*cosy
                  if (abs(dyb).lt.wprof_ik(ir,KKK)) then
                     mask3(i,j) = mask3(i,j) + sprof_ik(ir,KKK)
                     endif
                  endif
               mask0(i,j) = mask1(i,j) + mask2(i,j) + mask3(i,j)
               enddo
            ia1(j) = 2001
            ia2(j) = 2001
            ia3(j) = 2001
            ib1(j) = 0001
            ib2(j) = 0001
            ib3(j) = 0001
            do i = 0001, 2001
               if (mask1(i,j).gt.0) ib1(j) = i
               if (mask2(i,j).gt.0) ib2(j) = i
               if (mask3(i,j).gt.0) ib3(j) = i
               enddo
            do i = 2001, 0001, -1
               if (mask1(i,j).gt.0) ia1(j) = i
               if (mask2(i,j).gt.0) ia2(j) = i
               if (mask3(i,j).gt.0) ia3(j) = i
               enddo
            if (ia1(j).ne.2001) then
               ja1 = min(j,ja1)
               jb1 = max(j,jb1)
               endif
            if (ia2(j).ne.2001) then
               ja2 = min(j,ja2)
               jb2 = max(j,jb2)
               endif
            if (ia3(j).ne.2001) then
               ja3 = min(j,ja3)
               jb3 = max(j,jb3)
               endif
            enddo
 
         iraw = int(xraw+0.5)
         jraw = int(yraw+0.5)
         iu = iraw
         ju = jraw
 
         if (iu+1000.le. 0001 ) goto 333
         if (ju+1000.le. 0001 ) goto 333
         if (iu-1000.ge.4096) goto 333
         if (ju-1000.ge.4096) goto 333
 
         do j = ja2, jb2
            jj = ju + (j-1001)
            if (jj.ge.001.and.jj.le.4096) then
               do i = ia2(j), ib2(j)
                  ii = iu + (i-1001)
                  if (ii.ge.001.and.ii.le.4096) then
                     pixm_sum(ii,jj)= pixm_sum(ii,jj)+(zu*mask2(i,j))**2
                     pixm_max(ii,jj)= max(pixm_max(ii,jj),zu*mask2(i,j))
                     endif
                  enddo
               endif
            enddo
 
         do j = ja3, jb3
            jj = ju + (j-1001)
            if (jj.ge.001.and.jj.le.4096) then
               do i = ia3(j), ib3(j)
                  ii = iu + (i-1001)
                  if (ii.ge.001.and.ii.le.4096) then
                     pixm_sum(ii,jj)= pixm_sum(ii,jj)+(zu*mask3(i,j))**2
                     pixm_max(ii,jj)= max(pixm_max(ii,jj),zu*mask3(i,j))
                     endif
                  enddo
               endif
            enddo
 
c        if (iu+iref_bb(BB).lt. 0001 ) goto 333
c        if (iu-iref_bb(BB).gt.4096) goto 333
c        if (ju+jref_bb(BB).lt. 0001 ) goto 333
c        if (ju-jref_bb(BB).gt.4096) goto 333
 
         do j = ja1, jb1
            jj = ju + (j-1001)
            if (jj.ge.001.and.jj.le.4096) then
               do i = ia1(j), ib1(j)
                  ii = iu + (i-1001)
                  if (ii.ge.001.and.ii.le.4096) then
                     pixm_sum(ii,jj)= pixm_sum(ii,jj)+(zu*mask1(i,j))**2
                     pixm_max(ii,jj)= max(pixm_max(ii,jj),zu*mask1(i,j))
                     endif
                  enddo
               endif
            enddo
 
  333    continue
         enddo
 
      do i = 0001, 4096
      do j = 0001, 4096
         pixm(i,j) = sqrt(pixm_sum(i,j))
         enddo
         enddo
 
      return
      end
 
 
 
 
c------------------------------------------------------
c
c this will act like a simple array
c
      real function rprof_ik(ir,k)
      implicit none
 
      integer ir, k
      real rprof_arr(1001,10)
      common /rprof_arr_/rprof_arr
 
      integer dgen(23)
      common /dgen_/dgen
 
      real rgen(10,23)
      common /rgen_/rgen
 
      logical first_rprof
      common /first_rprof_/ first_rprof
      data first_rprof/.true./
 
      integer i, ii, iiu
      integer kk
      real    ff
 
      if (first_rprof) then
         do i = 0001, 1001
            iiu = 0
            do ii = 1, 22
               if (i.ge.dgen(ii)) iiu = ii
               enddo
            if (iiu.eq.00) stop 'screech iiu.eq.00'
            if (iiu.ge.23) stop 'screech iiu.ge.23'
            ff = (i-dgen(iiu))*1.0/(dgen(iiu+1)-dgen(iiu))
            do kk = 01, 10
               rprof_arr(i,kk) = rgen(kk,iiu)
     .                         + ff*(rgen(kk,iiu+1)-rgen(kk,iiu))
               enddo
            enddo
         first_rprof = .false.
         endif
 
      rprof_ik = 0.
      if (ir.lt.0001.or.ir.gt.1001) return
      rprof_ik = rprof_arr(ir,k)
 
      return
      end
 
 
 
c------------------------------------------------------
c
c this will act like a simple array ; the spike profile
c
      real function sprof_ik(ir,k)
      implicit none
 
      integer ir, k
      real sprof_arr(1001,10)
      common /sprof_arr_/sprof_arr
 
      integer dgen(23)
      common /dgen_/dgen
 
      real sgen(10,23)
      common /sgen_/sgen
 
      logical first_sprof
      common /first_sprof_/ first_sprof
      data first_sprof/.true./
 
      integer i, ii, iiu
      integer kk
      real    ff
 
      if (first_sprof) then
         do i = 0001, 1001
            iiu = 0
            do ii = 1, 22
               if (i.ge.dgen(ii)) iiu = ii
               enddo
            if (iiu.eq.00) stop 'screech iiu.eq.00'
            if (iiu.ge.23) stop 'screech iiu.ge.23'
            ff = (i-dgen(iiu))*1.0/(dgen(iiu+1)-dgen(iiu))
            do kk = 01, 10
               sprof_arr(i,kk) = sgen(kk,iiu)
     .                         + ff*(sgen(kk,iiu+1)-sgen(kk,iiu))
               enddo
            enddo
         first_sprof = .false.
         endif
 
      sprof_ik = 0.
      if (ir.lt.0001.or.ir.gt.1001) return
      sprof_ik = sprof_arr(ir,k)
 
      return
      end
 
 
 
c------------------------------------------------------
c
c this will act like a simple array ; the spike-width profile
c
      integer function wprof_ik(ir,k)
      implicit none
 
      integer ir, k
      integer wprof_arr(1001,10)
      common /wprof_arr_/wprof_arr
 
      integer dgen(23)
      common /dgen_/dgen
 
      integer wgen(10,23)
      common /wgen_/wgen
 
      logical first_wprof
      common /first_wprof_/ first_wprof
      data first_wprof/.true./
 
      integer i, ii, iiu
      integer kk
      real    ff
 
      if (first_wprof) then
         do i = 0001, 1001
            iiu = 0
            do ii = 1, 22
               if (i.ge.dgen(ii)) iiu = ii
               enddo
            if (iiu.eq.00) stop 'screech iiu.eq.00'
            if (iiu.ge.23) stop 'screech iiu.ge.23'
            ff = (i-dgen(iiu))*1.0/(dgen(iiu+1)-dgen(iiu))
            do kk = 01, 10
               wprof_arr(i,kk) = wgen(kk,iiu)
     .                         + ff*(wgen(kk,iiu+1)-wgen(kk,iiu)) + 0.5
               enddo
            enddo
         first_wprof = .false.
         endif
 
      wprof_ik = 0.
      if (ir.lt.0001.or.ir.gt.1001) return
      wprof_ik = wprof_arr(ir,k)
 
      return
      end
 
 
 
c-------------------------------------------------
c this regulates the radial parametrization
c
      block data bd_dgen
      integer dgen(23)
      common /dgen_/dgen
      data dgen/
     . 0000, 0005, 0010, 0015, 0020, 0030, 0040, 0050, 0060, 0075,
     . 0100, 0125, 0150, 0200, 0250, 0300, 0400, 0500, 0600, 0700,
     . 0800, 0900, 1000/
      end
 
 
c-------------------------------------------------
c this regulates the radial profile
c
      block data bd_rgen
      real rgen(10,23) ! profile for radial part
      common /rgen_/rgen
      data rgen/
c       PC1  WF2   WF3   WF4   WFCb  WFCt  HRC   WFC3 WFC3IR
c       1     2     3     4     5     6     7     8     9   10
     . 0e-00,0e-00,0e-00,0e-00,8e-04,0e-00,6e-03,1e-03,7e-04,0e-00, ! 000 --- 01
     . 0e-00,0e-00,0e-00,0e-00,4e-04,0e-00,3e-03,1e-03,5e-04,0e-00, ! 005 --- 02
     . 0e-00,0e-00,0e-00,0e-00,2e-04,0e-00,2e-03,5e-04,7e-05,0e-00, ! 010 --- 03
     . 0e-00,0e-00,0e-00,0e-00,4e-05,0e-00,6e-04,2e-04,3e-05,0e-00, ! 015 --- 04
     . 0e-00,0e-00,0e-00,0e-00,1e-05,0e-00,3e-04,1e-05,1e-05,0e-00, ! 020 --- 05
     . 0e-00,0e-00,0e-00,0e-00,5e-06,0e-00,4e-05,6e-06,4e-06,0e-00, ! 030 --- 06
     . 0e-00,0e-00,0e-00,0e-00,2e-06,0e-00,8e-06,5e-06,2e-06,0e-00, ! 040 --- 07
     . 0e-00,0e-00,0e-00,0e-00,1e-06,0e-00,1e-06,4e-06,1e-06,0e-00, ! 050 --- 08
     . 0e-00,0e-00,0e-00,0e-00,8e-07,0e-00,0e-00,2e-06,6e-07,0e-00, ! 060 --- 09
     . 0e-00,0e-00,0e-00,0e-00,3e-07,0e-00,0e-00,8e-07,2e-08,0e-00, ! 075 --- 10
     . 0e-00,0e-00,0e-00,0e-00,1e-07,0e-00,0e-00,2e-07,1e-08,0e-00, ! 100 --- 11
     . 0e-00,0e-00,0e-00,0e-00,0e-00,0e-00,0e-00,1e-07,0e-00,0e-00, ! 125 --- 12
     . 0e-00,0e-00,0e-00,0e-00,0e-00,0e-00,0e-00,6e-08,0e-00,0e-00, ! 150 --- 13
     . 0e-00,0e-00,0e-00,0e-00,0e-00,0e-00,0e-00,3e-08,0e-00,0e-00, ! 200 --- 14
     . 0e-00,0e-00,0e-00,0e-00,0e-00,0e-00,0e-00,1e-08,0e-00,0e-00, ! 250 --- 15
     . 0e-00,0e-00,0e-00,0e-00,0e-00,0e-00,0e-00,0e-00,0e-00,0e-00, ! 300 --- 16
     . 0e-00,0e-00,0e-00,0e-00,0e-00,0e-00,0e-00,0e-00,0e-00,0e-00, ! 400 --- 17
     . 0e-00,0e-00,0e-00,0e-00,0e-00,0e-00,0e-00,0e-00,0e-00,0e-00, ! 500 --- 18
     . 0e-00,0e-00,0e-00,0e-00,0e-00,0e-00,0e-00,0e-00,0e-00,0e-00, ! 600 --- 19
     . 0e-00,0e-00,0e-00,0e-00,0e-00,0e-00,0e-00,0e-00,0e-00,0e-00, ! 700 --- 20
     . 0e-00,0e-00,0e-00,0e-00,0e-00,0e-00,0e-00,0e-00,0e-00,0e-00, ! 800 --- 21
     . 0e-00,0e-00,0e-00,0e-00,0e-00,0e-00,0e-00,0e-00,0e-00,0e-00, ! 900 --- 22
     . 0e-00,0e-00,0e-00,0e-00,0e-00,0e-00,0e-00,0e-00,0e-00,0e-00/ !1000 --- 23
      end
 
 
c-------------------------------------------------
c this regulates the spike profile
c
      block data bd_sgen
      real sgen(10,23) ! profile for spike
      common /sgen_/sgen
      data sgen/
c       1     2     3     4     5     6     7     8     9   10
     . 0e-00,0e-00,0e-00,0e-00,0e-05,0e-00,0e-03,0e-04,0e-04,0e-00, ! 000 --- 01
     . 0e-00,0e-00,0e-00,0e-00,4e-05,0e-00,2e-04,4e-05,2e-04,0e-00, ! 005 --- 02
     . 0e-00,0e-00,0e-00,0e-00,8e-05,0e-00,3e-04,7e-05,3e-04,0e-00, ! 010 --- 03
     . 0e-00,0e-00,0e-00,0e-00,5e-05,0e-00,1e-04,7e-05,2e-04,0e-00, ! 015 --- 04
     . 0e-00,0e-00,0e-00,0e-00,3e-05,0e-00,5e-05,5e-05,1e-04,0e-00, ! 020 --- 05
     . 0e-00,0e-00,0e-00,0e-00,2e-05,0e-00,4e-05,4e-05,6e-05,0e-00, ! 030 --- 06
     . 0e-00,0e-00,0e-00,0e-00,1e-05,0e-00,3e-05,3e-05,3e-05,0e-00, ! 040 --- 07
     . 0e-00,0e-00,0e-00,0e-00,1e-05,0e-00,2e-05,2e-05,1e-05,0e-00, ! 050 --- 08
     . 0e-00,0e-00,0e-00,0e-00,8e-06,0e-00,1e-05,1e-05,8e-06,0e-00, ! 060 --- 09
     . 0e-00,0e-00,0e-00,0e-00,4e-06,0e-00,5e-06,8e-06,6e-06,0e-00, ! 075 --- 10
     . 0e-00,0e-00,0e-00,0e-00,2e-06,0e-00,0e-00,6e-06,5e-06,0e-00, ! 100 --- 11
     . 0e-00,0e-00,0e-00,0e-00,1e-06,0e-00,0e-00,4e-06,3e-06,0e-00, ! 125 --- 12
     . 0e-00,0e-00,0e-00,0e-00,3e-07,0e-00,0e-00,2e-06,2e-06,0e-00, ! 150 --- 13
     . 0e-00,0e-00,0e-00,0e-00,2e-07,0e-00,0e-00,1e-06,1e-06,0e-00, ! 200 --- 14
     . 0e-00,0e-00,0e-00,0e-00,1e-07,0e-00,0e-00,6e-07,6e-07,0e-00, ! 250 --- 15
     . 0e-00,0e-00,0e-00,0e-00,4e-08,0e-00,0e-00,3e-07,4e-07,0e-00, ! 300 --- 16
     . 0e-00,0e-00,0e-00,0e-00,1e-08,0e-00,0e-00,1e-07,2e-07,0e-00, ! 400 --- 17
     . 0e-00,0e-00,0e-00,0e-00,4e-09,0e-00,0e-00,5e-08,1e-07,0e-00, ! 500 --- 18
     . 0e-00,0e-00,0e-00,0e-00,1e-09,0e-00,0e-00,3e-08,5e-08,0e-00, ! 600 --- 19
     . 0e-00,0e-00,0e-00,0e-00,0e-00,0e-00,0e-00,3e-08,4e-08,0e-00, ! 700 --- 20
     . 0e-00,0e-00,0e-00,0e-00,0e-00,0e-00,0e-00,3e-08,3e-08,0e-00, ! 800 --- 21
     . 0e-00,0e-00,0e-00,0e-00,0e-00,0e-00,0e-00,2e-08,2e-08,0e-00, ! 900 --- 22
     . 0e-00,0e-00,0e-00,0e-00,0e-00,0e-00,0e-00,1e-08,1e-08,0e-00/ !1000 --- 23
 
 
 
 
      end
 
 
c-------------------------------------------------
c this regulates the spike width
c
      block data bd_wgen
      integer wgen(10,23) ! spike width
      common /wgen_/wgen
      data wgen/
c
c       1  2  3  4  5  6  7  8  9 10
c
     .  5, 5, 5, 5, 5, 5, 5, 5, 5, 5, ! 000 --- 01
     .  5, 5, 5, 5, 5, 5, 5, 5, 5, 5, ! 005 --- 02
     .  5, 5, 5, 5, 5, 5, 5, 5, 5, 5, ! 010 --- 03
     .  5, 5, 5, 5, 5, 5, 5, 5, 5, 5, ! 015 --- 04
     .  5, 5, 5, 5, 4, 5, 5, 4, 5, 5, ! 020 --- 05
     .  5, 5, 5, 5, 4, 5, 5, 4, 5, 5, ! 030 --- 06
     .  5, 5, 5, 5, 3, 5, 5, 3, 5, 5, ! 040 --- 07
     .  5, 5, 5, 5, 3, 5, 5, 3, 5, 5, ! 050 --- 08
     .  5, 5, 5, 5, 3, 5, 5, 3, 5, 5, ! 060 --- 09
     .  5, 5, 5, 5, 3, 5, 5, 3, 5, 5, ! 075 --- 10
     .  5, 5, 5, 5, 3, 5, 5, 3, 5, 5, ! 100 --- 11
     .  5, 5, 5, 5, 3, 5, 5, 3, 5, 5, ! 125 --- 12
     .  5, 5, 5, 5, 3, 5, 5, 3, 5, 5, ! 150 --- 13
     .  5, 5, 5, 5, 3, 5, 5, 3, 6, 5, ! 200 --- 14
     .  5, 5, 5, 5, 3, 5, 5, 3, 6, 5, ! 250 --- 15
     .  5, 5, 5, 5, 3, 5, 5, 3, 8, 5, ! 300 --- 16
     .  5, 5, 5, 5, 4, 5, 5, 2,10, 5, ! 400 --- 17
     .  5, 5, 5, 5, 4, 5, 5, 2,11, 5, ! 500 --- 18
     .  5, 5, 5, 5, 4, 5, 5, 2,12, 5, ! 600 --- 19
     .  5, 5, 5, 5, 5, 5, 5, 2,13, 5, ! 700 --- 20
     .  5, 5, 5, 5, 5, 5, 5, 2,14, 5, ! 800 --- 21
     .  5, 5, 5, 5, 5, 5, 5, 2,15, 5, ! 900 --- 22
     .  5, 5, 5, 5, 5, 5, 5, 2,15, 5/ !1000 --- 23
      end
 
 
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/ROUTINES/WFC.09x10/find_xyzXX_NAXIS.f"
c****                                         
c*********************************************
                                              
 
c----------------------------------------------
c
c uses a chisq minimization technique to find the optimal
c values for (x,y,z) using the inner 5x5 pixels; it
c takes the sky value (SKIN) as a given.  It does a very
c crude grid search to find the chisq minimim.
c
c HOWWT = 0  flat wgt
c HOWWT = 1  S/N wtg
c
c
      subroutine find_xyzXX_NAXIS(x,y,z,SKIN,qfit,pix,psfloc,
     .                            NAXIS1,NAXIS2,HOWWT)
      implicit none
 
      real*8 x,y,z
 
      real*8 SKIN
      real    qfit
 
      integer     NAXIS1,NAXIS2
      real    pix(NAXIS1,NAXIS2)
      real    psfloc(101,101)
      integer HOWWT
 
      real psfx(5,5)
      real pixx(5,5)
      real sig2(5,5)
      real psfx_(5,5)
 
      integer mmeth
      common / mmeth / mmeth
 
      integer NIT
      integer ii, jj, i, j
 
      real rpsf_phot
      real ftot
      real ptot
      real ftotw
      real ptotw
 
      real fxu, fx0, fxout
      real fyu, fy0, fyout
      real  zu,  zout
      real  dx, dy
 
      real  dd(11)
      data dd /0.300,0.200,0.100,0.050,0.030,
     .         0.020,0.010,0.005,0.003,0.002,
     .         0.001/
 
      real dxn(9)
      real dyn(9)
      data dxn/0.000,+1.000,-1.000,+0.000,+0.000,
     .               +0.707,+0.707,-0.707,-0.707/
      data dyn/0.000,+0.000,+0.000,+1.000,-1.000,
     .               +0.707,-0.707,-0.707,+0.707/
 
      integer nnn
 
      real err, chimin, chitot
      real abstot, absmin
      real pixtot
      real pixval
 
      integer NDONE
      data NDONE/0/
 
      mmeth = 5
 
      ii = int(x+0.5)
      jj = int(y+0.5)
 
      z = 0.
      qfit = 0.99
 
      if (ii.lt. 0001 +2) return
      if (jj.lt. 0001 +2) return
      if (ii.gt.NAXIS1-2) return
      if (jj.gt.NAXIS2-2) return
 
      if (HOWWT.ne.0.and.HOWWT.ne.1) then
         print*,' '
         print*,'find_xymXX_chisq_NAXIS illegal param: '
         print*,'   HOWWT = ',HOWWT
         print*,' '
         stop
         endif
 
      pixtot = 0.
      do i = 1, 5
      do j = 1, 5
         pixval    = pix(ii+i-3,jj+j-3)
         pixx(i,j) = pixval - SKIN
         pixtot = pixtot + pixx(i,j)
         sig2(i,j) = max(pixval,0.00) + 25.00
         enddo
         enddo
 
      fxout = x-ii
      fyout = y-jj
      do NIT = 01, 11
         fx0 = fxout
         fy0 = fyout
         chimin = 9e30
         do nnn = 01, 09
            fxu = fx0 + dd(NIT)*dxn(nnn)
            fyu = fy0 + dd(NIT)*dyn(nnn)
            ftot  = 0.
            ptot  = 0.
            ftotw = 0.
            ptotw = 0.
            do i = 1, 5
            do j = 1, 5
               dx = (i-3) - fxu
               dy = (j-3) - fyu
               psfx(i,j) = rpsf_phot(dx,dy,psfloc)
               ftot  = ftot  + psfx(i,j)
               ptot  = ptot  + pixx(i,j)
               ftotw = ftotw + psfx(i,j)*psfx(i,j)/sig2(i,j)
               ptotw = ptotw + pixx(i,j)*psfx(i,j)/sig2(i,j)
               enddo
               enddo
            zu = ptot/ftot
            chitot = 0.
            abstot = 0.
            do i = 1, 5
            do j = 1, 5
               err  = abs(pixx(i,j)-zu*psfx(i,j))
               if (HOWWT.eq.0) chitot = chitot + err**2
               if (HOWWT.eq.1) chitot = chitot + err**2/sig2(i,j)
               abstot = abstot + err
               enddo
               enddo
            if (chitot.lt.chimin) then
               chimin = chitot
               fxout = fxu
               fyout = fyu
               zout  =  zu
               absmin = abstot/pixtot
               do i = 1, 5
               do j = 1, 5
                  psfx_(i,j) = psfx(i,j)
                  enddo
                  enddo
               endif
            enddo!nnn
         enddo!NIT
 5    continue
 
      x = ii + fxout
      y = jj + fyout
      z = zout
 
      qfit = sqrt(chitot/(25-3))
      qfit = absmin
      if (qfit.gt.9.99) qfit = 9.99
 
      return
      end
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/ROUTINES/WFC.09x10/z_xyoptXX_NAXIS.f"
c****                                         
c*********************************************
                                              
 
c-----------------------------------------------
c
 
      real function z_xyoptXX_NAXIS(xin,yin,skin,pix,RADFIT,
     .                              psfloc,
     .                              NAXIS1,NAXIS2,DOSNWT)
      implicit none
      real*8 xin
      real*8 yin
      real*8 skin
      integer NAXIS1
      integer NAXIS2
      real    pix(NAXIS1,NAXIS2)
      real    RADFIT
      real    psfloc(101,101)
      integer DOSNWT
 
      real   rpsf_phot
 
      real    dx, dy
      integer ix, iy
      integer i,  j
 
      integer NL, NLs
      real dl(500)
      real pl(500)
      real fl(500)
      real sl(500)
      real wl(500)
 
      real*8 sumpf
      real*8 sumff
      real*8 sumpfw
      real*8 sumffw
      real zuse
 
      integer NARG
      common /NARG_/NARG
      integer mmeth
      common /mmeth/ mmeth
 
 
      ix = int(xin+0.5)
      iy = int(yin+0.5)
 
      z_xyoptXX_NAXIS = 0.
 
      if (ix.le. 0001 +RADFIT) return
      if (iy.le. 0001 +RADFIT) return
      if (ix.ge.NAXIS1+RADFIT) return
      if (iy.ge.NAXIS2+RADFIT) return
 
      NLs = 0
      do i = ix-int(RADFIT+1),ix+int(RADFIT+1)
      do j = iy-int(RADFIT+1),iy+int(RADFIT+1)
         dx = i-xin
         dy = j-yin
         if (sqrt(dx**2+dy**2).lt.RADFIT) then
            NLs = NLs + 1
            if (NLs.gt.500) then
               print*,'z_opt: NLs.gt.500'
               stop
               endif
            pl(NLs) = pix(i,j)-skin
            fl(NLs) = rpsf_phot(dx,dy,psfloc)
            dl(NLs) = sqrt(dx**2+dy**2)
            endif
         enddo
         enddo
 
      sumpf = 0.
      sumff = 0.
      do NL = 01, NLs
         sumpf = sumpf + fl(NL)*pl(NL)
         sumff = sumff + fl(NL)*fl(NL)
         enddo
      zuse = sumpf/sumff
      if (zuse.lt.0) zuse = 0.
 
      do NL = 1, NLs
         sl(NL) = sqrt(zuse*fl(NL)+skin+15)
         wl(NL) = 1.00/sl(NL)**2
         if (DOSNWT.eq.0) wl(NL) = 1.0
         enddo
 
      sumpfw = 0.
      sumffw = 0.
      do NL = 01, NLs
         sumpfw = sumpfw + fl(NL)*pl(NL)*wl(NL)
         sumffw = sumffw + fl(NL)*fl(NL)*wl(NL)
         enddo
      zuse = sumpfw/sumffw
      if (zuse.lt.0) zuse = 0.
 
      z_xyoptXX_NAXIS = zuse
 
      return
      end
 
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/FITSIO/PUTCHAR/img2putchar_r4.f"
c****                                         
c*********************************************
                                              
 
      subroutine img2putstr_r4(string,NELEM,val,ic,jc,pix,NX,NY)
      implicit none
      integer   NELEM
      character STRING(NELEM)
      integer ic,jc
      real*4  val
      integer NX,NY
      real*4  pix(NX,NY)
      integer N
 
      do N = 1, NELEM
         call img2putchar_r4(string(N),val,ic+(N-1)*6,jc,pix,NX,NY)
         enddo
 
      return
      end
 
 
 
      subroutine img2putchar_r4(putchar,val,ic,jc,pix,NX,NY)
      implicit  none
      character putchar
      integer   ic,jc
      real*4    val
      integer   NX,NY
      real*4    pix(NX,NY)
 
      integer i,j,n,nchar
 
      character*27 INFO(38)
      data INFO/
     .     'aA.***.*...*******...**...*',
     .     'bB****.*..******.*..******.',
     .     'cC.*****....*....*.....****',
     .     'dD****.*...**...**...*****.',
     .     'eE******....***..*....*****',
     .     'fF******....***..*....*....',
     .     'gG******....*..***...******',
     .     'hH*...**...*******...**...*',
     .     'iI.***...*....*....*...***.',
     .     'jJ....*....*....**...*.***.',
     .     'kK*...**..*.***..*..*.*...*',
     .     'lL*....*....*....*....*****',
     .     'mM*...***.***.*.**...**...*',
     .     'nN*...***..**.*.**..***...*',
     .     'oO.***.*...**...**...*.***.',
     .     'pP******...*******....*....',
     .     'qQ******...**...**..*.***.*',
     .     'rR******...****..*..*.*...*',
     .     'sS******....*****....******',
     .     'tT*****..*....*....*....*..',
     .     'uU*...**...**...**...******',
     .     'vV*...**...**...*.*.*...*..',
     .     'wW*...**...**.*.***.***...*',
     .     'xX*...*.*.*...*...*.*.*...*',
     .     'yY*...*.*.*...*....*....*..',
     .     'zZ*****...*...*...*...*****',
     .     '00.***.*...**.*.**...*.***.',
     .     '11.**....*....*....*...***.',
     .     '22*****....*******....*****',
     .     '33*****....******....******',
     .     '44...*.*..*.*****...*....*.',
     .     '55******....*****....******',
     .     '66******....******...******',
     .     '77*****....*...*....*....*.',
     .     '88******...*******   ******',
     .     '99******...******.   ******',
     .     '  .........................',
     .     '........................*..'/
 
      do N = 1, 38
         if (putchar.eq.INFO(N)(1:1).or.
     .       putchar.eq.INFO(N)(2:2)) then
            do i = -2,2
            do j = -2,2
               nchar = 3 + i+2 + (2-j)*5
               if (INFO(N)(nchar:nchar).eq.'*')
     .            pix(ic+i,jc+j) = val
               enddo
               enddo
            endif
         enddo
 
      return
      end
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/ROUTINES/SORT/quiksort.f"
c****                                         
c*********************************************
                                              
 
 
C From HDK@psuvm.psu.edu Thu Dec  8 15:27:16 MST 1994
C
C The following was converted from Algol recursive to Fortran iterative
C by a colleague at Penn State (a long time ago - Fortran 66, please
C excuse the GoTo's). The following code also corrects a bug in the
C Quicksort algorithm published in the ACM (see Algorithm 402, CACM,
C Sept. 1970, pp 563-567; also you younger folks who weren't born at
C that time might find interesting the history of the Quicksort
C algorithm beginning with the original published in CACM, July 1961,
C pp 321-322, Algorithm 64). Note that the following algorithm sorts
C integer data; actual data is not moved but sort is affected by sorting
C a companion index array (see leading comments). The data type being
C sorted can be changed by changing one line; see comments after
C declarations and subsequent one regarding comparisons(Fortran
C 77 takes care of character comparisons of course, so that comment
C is merely historical from the days when we had to write character
C compare subprograms, usually in assembler language for a specific
C mainframe platform at that time). But the following algorithm is
C good, still one of the best available.
 
 
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
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/FITSIO/GEN/query_hdre.f"
c****                                         
c*********************************************
                                              
 
c---------------------------------------------------
c
c This routine checks to see if the header of an image
c contains a particular keyword. If it does, it returns
c its value-string in STREAMX.
c
 
      subroutine query_hdre(filename,FIELDX,streamx,NEXTu)
      implicit none
 
      character*(*) filename
 
      character*8  field
      character*20 stream
 
      integer NEXTu
 
      character*8   fieldx
      character*20  streamx
      character*200 FILENAMEu
 
      integer  i
      integer ios, k
 
      character*2880 buff
 
      integer NREAD
      logical EXTEND
 
      integer BITPIX
      integer NAXIS1, NAXIS2, NAXIS3
 
c-----------------------------------------------
 
      streamx = ' '
 
      do i = 1, 195
         if (FILENAME(i:i+4).eq.'.fits') goto 3
         enddo
      print*,'query_hdre...',FILENAME
      stop 'no ".fits" '
 
    3 continue
      FILENAMEu = FILENAME(1:i+4)
 
      close(10)
      open(10,file=FILENAMEu,status='old',iostat=ios,
     .  err=900,recl=2880,form='UNFORMATTED',
     .  access='DIRECT')
 
      streamx = 'NULL'
      i = 0
      NREAD  = 0
      BITPIX = 8
      EXTEND = .false.
 100  continue
      i = i + 1
      read(10,rec=i,iostat=ios) buff
      if (ios.eq.5002) then ! bad option (?); maybe end of file?
          return
          endif
      if (ios.lt.   0) goto 901
      NAXIS1 = 0
      NAXIS2 = 1
      NAXIS3 = 1
      do k = 0, 35, 1
         field  = buff(k*80+01:k*80+08)
         stream = buff(k*80+11:k*80+31)
         if (field.eq.fieldx) then
            streamx = stream(1:20)
            if (NEXTu.eq.-1) then
                close(10)
                return
                endif
            endif
         if (field.eq.'EXTEND  ') read(stream,*) EXTEND
         if (field.eq.'END     ') goto 101
         if (field.eq.'BITPIX  ') read(stream,*) BITPIX
         if (field.eq.'NAXIS1  ') read(stream,*) NAXIS1
         if (field.eq.'NAXIS2  ') read(stream,*) NAXIS2
         if (field.eq.'NAXIS3  ') read(stream,*) NAXIS3
 109     continue
         enddo
      goto 100
 101  continue
      if (NREAD.ne.NEXTu.or.NEXTu.eq.-1) then
          NREAD = NREAD + 1
          i = i + (abs(BITPIX)/8*NAXIS1*NAXIS2*NAXIS3+2879)/2880
          goto 100
          endif
      close(10)
      return
 
  900 continue
      print*,' '
      print*,'query_hdre ERROR EXIT. '
      print*,' '
      print*,'FILE OPEN ERROR: '
      print*,' '
      write(*,'(''       FIELDX: '',a8)') FIELDX
      print*,' '
      write(*,'(''PROBLEM FILE : '',a)') FILENAME
      write(*,'(''PROBLEM FILEu: '',a)') FILENAMEu
      print*,' '
      write(*,'(''          IOS: '',i5)') ios
      print*,' '
      print*,' '
      print*,' '
      stop
 
  901 continue
      print*,' '
      print*,'query_hdre ERROR EXIT. '
      print*,'i/o error...'
      print*,' '
      write(*,'(''       FIELDX: '',a8)') FIELDX
      print*,' '
      write(*,'(''PROBLEM FILE : '',a)') FILENAME
      write(*,'(''PROBLEM FILEu: '',a)') FILENAMEu
      print*,' '
      write(*,'(''         IOS: '',i5)') ios
      print*,' '
      stop
      end
 
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/FITSIO/GEN/query_hdre_i4.f"
c****                                         
c*********************************************
                                              
 
c---------------------------------------------------
c
 
      integer function i4_query_hdre(filename,FIELDX,NEXTu)
      implicit none
 
      character*(*) FILENAME
      character*8   FIELDX
      integer       NEXTu
 
      integer i4
 
      call query_hdre_i4(FILENAME,FIELDX,i4,NEXTu)
 
      i4_query_hdre = i4
 
      return
      end
 
 
 
 
c---------------------------------------------------
c
 
 
 
      subroutine query_hdre_i4(filename,FIELDX,i4,NEXTu)
      implicit none
 
      character*(*) filename
 
      character*8  field
      integer*4    i4
      integer      NEXTu
 
      character*20 stream
 
      character*8  fieldx
      character*20 streamx
 
      integer  i
      integer ios, k
 
      character*2880 buff
 
      integer nread
      logical EXTEND
 
c-----------------------------------------------
 
      streamx = ' '
 
      call query_hdre(filename,FIELDX,streamx,NEXTu)
      read(streamx,*) i4
      return
 
      stop
      end
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/FITSIO/GEN/query_hdre_r4.f"
c****                                         
c*********************************************
                                              
 
 
      real*4 function r4_query_hdre(filename,FIELDX,NEXTu)
      implicit none
 
      character*(*) FILENAME
      character*8   FIELDX
      integer       NEXTu
 
      real          r4
 
      call query_hdre_r4(filename,FIELDX,r4,NEXTu)
 
      r4_query_hdre = r4
 
      return
      end
 
 
 
 
c---------------------------------------------------
c
c This routine checks to see if the header of an image
c contains a particular keyword. If it does, it returns
c its value-string in STREAMX.
c
 
      subroutine query_hdre_r4(filename,FIELDX,r4,NEXTu)
      implicit none
 
      character*(*) filename
      character*8   FIELDX
      real          r4
      integer       NEXTu
 
      character*20  streamx
 
 
      streamx = ' '
 
      call query_hdre(filename,FIELDX,streamx,NEXTu)
      read(streamx,*,err=999) r4
      return
 
  999 continue
      print*,'  '
      print*,'ERROR READING REAL IN query_hdre_r4...'
      print*,'  '
      print*,'FILENAME: ',FILENAME
      print*,'   FIELD: ',FIELDX
      print*,' STREAMX: ',STREAMX
      print*,'   NEXTu: ',NEXTu
      print*,'  '
      stop
      end
 
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/FITSIO/GEN/query_hdre_r8.f"
c****                                         
c*********************************************
                                              
 
 
      real*8 function r8_query_hdre(FILENAME,FIELDX,NEXTu)
      implicit none
 
      character*80 FILENAME
      character*8  FIELDX
      integer      NEXTu
 
      real*8 r8
 
 
      call query_hdre_r8(filename,FIELDX,r8,NEXTu)
 
      r8_query_hdre = r8
 
      return
      end
 
 
c---------------------------------------------------
c
c This routine checks to see if the header of an image
c contains a particular keyword. If it does, it returns
c its value-string in STREAMX.
c
 
      subroutine query_hdre_r8(filename,FIELDX,r8,NEXTu)
      implicit none
 
      character*(*) filename
      character*8   FIELDX
      real*8        r8
      integer       NEXTu
 
      character*20  streamx
 
 
      streamx = ' '
 
      call query_hdre(filename,FIELDX,streamx,NEXTu)
      read(streamx,*) r8
      return
 
      end
 
 
 
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/2MASS/sub_query_2mass.f"
c****                                         
c*********************************************
                                              
 
c--------------------------------------------------------
c
c query_loc_2MASS.F
c
c This routine will allow super-fast searching
c of a condensed 2MASS data base.  The condensed
c data base contains RA, DEC, MJ, MH, MK and the
c quality flags for each band.  It has 13 bytes
c for each star.  With 475M stars, the total size
c of the database is 5.7G.  It can be compressed to
c fit on a single DVD for storage or export.
c
c This routine will hardcode the routine to look
c for the data base on central
c
c Written by Jay Anderson, March 2011
c
c The original data comes from the websites associated
c with:
c
c     SKRUTSKIE et al 2006 AJ 131 1163
c
c
c--------------------------------------------------------
c
c
c
      subroutine query_loc_2MASS(RA_MIN,RA_MAX,DE_MIN,DE_MAX,
     .                           ra_o,de_o,mj_o,mh_o,mk_o,aa_o,Os)
      implicit none
 
      real*8      RA_MIN, RA_MAX
      real*8      DE_MIN, DE_MAX
      real*8      ra_o(9999)
      real*8      de_o(9999)
      real*8      mj_o(9999)
      real*8      mh_o(9999)
      real*8      mk_o(9999)
      character*3 aa_o(9999)
      integer Os
 
c
c---------------------------------------
c
 
      integer i0
      character*80 stringg
      character*80 STRING
 
      integer nstar
 
      real*8 ra
      real*8 de
      real jmag, jerr
      real hmag, herr
      real kmag, kerr
 
c---------------------------------------
c
c variables to hold the search-qualifying
c stars before we decide whether to output
c them
c
 
      integer     U
      integer     Us
 
      character*80 RA_STRING
      character*80 DE_STRING
      character*80 STRING_IN
 
 
c-----------------------------------
c
c read in one record at a time; this
c contains the current 277-star record
c
      byte   buff13_277(13,277)
 
      integer N
 
c-------------------------------------
c
c this map tells you what the first record
c is for any point in the sky
c
      integer NMAP_IJ(3601,1801)
 
 
c-------------------------------------
c
c these are the parameters that define
c the stars that quality for the search
c
c     RELATIVE-TYPE SEARCHES
c
 
      real*8 RCEN,  DCEN       ! (deg) the pivot RA/DEC (for radial searches)
      real   DRMIN, DRMAX      ! (deg) for radial searches or directed lists
      real   DXMAX, DYMAX      ! (deg) for localized box searches
 
      real   JCEN, HCEN, KCEN  ! the pivot JHK mags
      real   DJMIN, DJMAX      ! J mag lower and upper bounds (relative to list input)
      real   DHMIN, DHMAX      ! H mag ...
      real   DKMIN, DKMAX      ! K mag ...
 
c
c     ABSOLUTE-TYPE SEARCHES
c
      real*8 RMIN,  RMAX       ! (deg) min/max RA     (for box
      real*8 DMIN,  DMAX       ! (deg) min/max DEC     searches)
      real*8 EBMIN, EBMAX      ! (deg) min/max ecliptic longitude   (for box
      real*8 ELMIN, ELMAX      ! (deg) min/max ecliptic latitude     searches)
      real*8 GBMIN, GBMAX      ! (deg) min/max galactic longitude   (for box
      real*8 GLMIN, GLMAX      ! (deg) min/max galactic latitude     searches)
 
      real   JMIN, JMAX        ! absolute J mag lower/upper bounds
      real   HMIN, HMAX        ! absolute H mag lower/upper bounds
      real   KMIN, KMAX        ! absolute K mag lower/upper bounds
 
 
      real*8      dd_lcky
      real*8      ra_lcky, de_lcky
      real*8      mj_lcky, mh_lcky, mk_lcky
      character*3 aa_lcky
      real*8      dx_lcky, dy_lcky
 
 
      logical LCKY
      logical LIST
 
c---------------------------------------------------------------
 
      integer IIMINu,IIMAXu
      integer IIMIN, IIMAX
      integer JJMIN, JJMAX
 
      integer i, j
 
 
      real*8 rminij, rmaxij
      real*8 dminij, dmaxij
 
      byte usesky_x01( 360, 180)
      byte usesky_c01( 360, 180)
      byte usesky_x10(3600,1800)
      byte usesky_c10(3600,1800)
      byte use
      integer NUSE
 
      integer L, Ls
      integer jl(999)
      integer i1l(999)
      integer i2l(999)
 
      integer LL, LLs
      integer dn_ll(9999)
      integer jj_ll(9999)
      integer ii1_ll(9999)
      integer ii2_ll(9999)
      integer nn1_ll(9999)
      integer nn2_ll(9999)
 
      integer rec0, rec1, rec2, recu
      integer NN, nn1, nn2
 
      real*8 dx, dy, dd, rd2x, rd2y
      integer ii, ii1, ii2
      integer jj, jj1, jj2
      integer ii1u, ii2u
      integer iiu,  jju
 
      integer iii, jjj, iiiu, jjju
c
c---------------------------------------------------------------
c
      integer*4   ira, ide
      integer*2   imj, imh, imk
      character*3 aaa
 
      integer NARG, NARGs
      character*1 UorV
 
      RCEN = -999.9
      DCEN = -999.9
 
 
      RMIN = RA_MIN
      RMAX = RA_MAX
      DMIN = DE_MIN
      DMAX = DE_MAX
 
      RCEN = (RMIN+RMAX)/2.0
      DCEN = (DMIN+DMAX)/2.0
 
      if (RMIN.gt.RMAX) stop 'RMIN.gt.RMAX'
      if (DMIN.gt.DMAX) stop 'DMIN.gt.DMAX'
 
 
      EBMIN = -999.9
      EBMAX = 9999.9
      ELMIN = -999.9
      ELMAX = 9999.9
 
      GBMIN = -999.9
      GBMAX = 9999.9
      GLMIN = -999.9
      GLMAX = 9999.9
 
      JMIN =  -9.9
      JMAX =  99.9
      HMIN =  -9.9
      HMAX =  99.9
      KMIN =  -9.9
      KMAX =  99.9
 
      JCEN = 0.00
      HCEN = 0.00
      KCEN = 0.00
 
      DRMIN = -999.99
      DRMAX =  999.99
      DJMIN = -999.99
      DJMAX =  999.99
      DHMIN = -999.99
      DHMAX =  999.99
      DKMIN = -999.99
      DKMAX =  999.99
      DXMAX =  999.99
      DYMAX =  999.99
 
      LCKY = .false.
      LIST = .false.
      UorV = 'U'
 
 
 
c---------------------------------------------------------
c
c this will read in the map to the database
c
c     print*,'OPEN BASE.DATA...'
c     open(12,file='/Volumes/data1/query_2mass/BASE.DATA',status='old',
      open(12,file='/user/jayander/2MASS/BASE.DATA',status='old',
     .        recl= 25941604,
     .        form='UNFORMATTED',
     .        access='DIRECT')
      read(12,rec=1) NMAP_IJ
      close(12)
 
 
c---------------------------------------------------------
c
c now, open the file for actual use as the dB
c
      rec0  = 3601*1801*4 / 13
c     open(12,file='/Volumes/data1/query_2mass/BASE.DATA',status='old',
      open(12,file='/user/jayander/2MASS/BASE.DATA',status='old',
     .        recl=3601,form='UNFORMATTED',
     .        access='DIRECT')
 
 
 137  continue
 
      STRING_IN = ''
 
      do i = 001, 360
      do j = 001, 180
         usesky_x01(i,j) = 0
         enddo
         enddo
 
      i = 1 + (RCEN)
      j = 1 + (DCEN + 90)
      if (i.ge.001.and.i.le.360.and.
     .    j.ge.001.and.j.le.180) usesky_x01(i,j) = 1
 
      do i = max(001,int(1+RMIN   -1)), min(360,int(1+RMAX   +1))
      do j = max(001,int(1+DMIN+90-1)), min(180,int(1+DMAX+90+1))
         usesky_x01(i,j) = 1
         enddo
         enddo
 
c      do i = 001, 360
c      do j = 001, 180
c         rminij = (i-1)*1.0
c         rmaxij = (i  )*1.0
c         dminij = (j-1)*1.0 - 90.00
c         dmaxij = (j  )*1.0 - 90.00
c         usesky_x01(i,j) =
c     .       max(usesky_x01(i,j),
c     .           use(rminij,dminij,jmin,hmin,kmin,
c     .               rcen,dcen,jcen,hcen,kcen,
c     .                    drmin,drmax,
c     .                    djmin,djmax,dhmin,dhmax,dkmin,dkmax,
c     .                    dxmax,dymax,
c     .               rmin,rmax,dmin,dmax,
c     .               ebmin,ebmax,elmin,elmax,
c     .               gbmin,gbmax,glmin,glmax,
c     .               jmin,jmax,hmin,hmax,kmin,kmax),
c     .           use(rmaxij,dminij,jmin,hmin,kmin,
c     .               rcen,dcen,jcen,hcen,kcen,
c     .                    drmin,drmax,
c     .                    djmin,djmax,dhmin,dhmax,dkmin,dkmax,
c     .                    dxmax,dymax,
c     .               rmin,rmax,dmin,dmax,
c     .               ebmin,ebmax,elmin,elmax,
c     .               gbmin,gbmax,glmin,glmax,
c     .               jmin,jmax,hmin,hmax,kmin,kmax),
c     .           use(rminij,dmaxij,jmin,hmin,kmin,
c     .               rcen,dcen,jcen,hcen,kcen,
c     .                    drmin,drmax,
c     .                    djmin,djmax,dhmin,dhmax,dkmin,dkmax,
c     .                    dxmax,dymax,
c     .               rmin,rmax,dmin,dmax,
c     .               ebmin,ebmax,elmin,elmax,
c     .               gbmin,gbmax,glmin,glmax,
c     .               jmin,jmax,hmin,hmax,kmin,kmax),
c     .           use(rmaxij,dmaxij,jmin,hmin,kmin,
c     .               rcen,dcen,jcen,hcen,kcen,
c     .                    drmin,drmax,
c     .                    djmin,djmax,dhmin,dhmax,dkmin,dkmax,
c     .                    dxmax,dymax,
c     .               rmin,rmax,dmin,dmax,
c     .               ebmin,ebmax,elmin,elmax,
c     .               gbmin,gbmax,glmin,glmax,
c     .               jmin,jmax,hmin,hmax,kmin,kmax))
c         enddo
c         enddo
 
 
      do i = 001, 360
      do j = 001, 180
         usesky_c01(i,j) = 0
         do ii = i-1, i+1
         do jj = j-1, j+1
            iiu = ii
            if (iiu.eq.361) iiu = 001
            if (iiu.eq.000) iiu = 360
            jju = jj
            if (jju.eq.181) jju = 180
            if (jju.eq.000) jju = 001
            if (usesky_x01(iiu,jju).eq.1) usesky_c01(i,j) = 1
            enddo
            enddo
         enddo
         enddo
 
c     call writfits_b1('usesky_x01.fits',usesky_x01,360,180)
c     call writfits_b1('usesky_c01.fits',usesky_c01,360,180)
 
      do i = 0001, 3600
      do j = 0001, 1800
         usesky_x10(i,j) = 0
         usesky_c10(i,j) = 0
         enddo
         enddo
 
      i = 1 + (RCEN   )*10
      j = 1 + (DCEN+90)*10
      if (i.ge.001.and.i.le.3600.and.
     .    j.ge.001.and.j.le.1800) usesky_x10(i,j) = 1
 
      do i = max(001,int(1+10*(RMIN   )-1)),
     .       min(360,int(1+10*(RMAX   )+1))
      do j = max(001,int(1+10*(DMIN+90)-1)),
     .       min(180,int(1+10*(DMAX+90)+1))
         usesky_x10(i,j) = 1
         enddo
         enddo
 
 
      JJMIN =  9999
      JJMAX = -9999
      IIMIN =  9999
      IIMAX = -9999
      do i = 001, 360
      do j = 001, 180
         ii1 = 1 + (i-1)*10 -1
         ii2 = 1 + (i  )*10 +1
         jj1 = 1 + (j-1)*10 -1
         jj2 = 1 + (j  )*10 +1
         if (usesky_c01(i,j).eq.1) then
            do jj = jj1, jj2
               jju = jj
               if (jju.lt.0001) jju = 0001
               if (jju.gt.1800) jju = 1800
               do ii = ii1, ii2
                  iiu = ii
                  if (iiu.lt.0001) iiu = iiu+3600
                  if (iiu.gt.3600) iiu = iiu-3600
                  rminij = (iiu-1)*0.1
                  rmaxij = (iiu  )*0.1
                  dminij = (jju-1)*0.1 - 90.00
                  dmaxij = (jju  )*0.1 - 90.00
                  usesky_x10(iiu,jju) =
     .                max(usesky_x10(iiu,jju),
     .                    use(rminij,dminij,jmin,hmin,kmin,
     .                        rcen,dcen,jcen,hcen,kcen,
     .                             drmin,drmax,
     .                             djmin,djmax,dhmin,dhmax,dkmin,dkmax,
     .                             dxmax,dymax,
     .                        rmin,rmax,dmin,dmax,
     .                        ebmin,ebmax,elmin,elmax,
     .                        gbmin,gbmax,glmin,glmax,
     .                        jmin,jmax,hmin,hmax,kmin,kmax),
     .                    use(rmaxij,dminij,jmin,hmin,kmin,
     .                        rcen,dcen,jcen,hcen,kcen,
     .                             drmin,drmax,
     .                             djmin,djmax,dhmin,dhmax,dkmin,dkmax,
     .                             dxmax,dymax,
     .                        rmin,rmax,dmin,dmax,
     .                        ebmin,ebmax,elmin,elmax,
     .                        gbmin,gbmax,glmin,glmax,
     .                        jmin,jmax,hmin,hmax,kmin,kmax),
     .                    use(rminij,dmaxij,jmin,hmin,kmin,
     .                        rcen,dcen,jcen,hcen,kcen,
     .                             drmin,drmax,
     .                             djmin,djmax,dhmin,dhmax,dkmin,dkmax,
     .                             dxmax,dymax,
     .                        rmin,rmax,dmin,dmax,
     .                        ebmin,ebmax,elmin,elmax,
     .                        gbmin,gbmax,glmin,glmax,
     .                        jmin,jmax,hmin,hmax,kmin,kmax),
     .                    use(rmaxij,dmaxij,jmin,hmin,kmin,
     .                        rcen,dcen,jcen,hcen,kcen,
     .                             drmin,drmax,
     .                             djmin,djmax,dhmin,dhmax,dkmin,dkmax,
     .                             dxmax,dymax,
     .                        rmin,rmax,dmin,dmax,
     .                        ebmin,ebmax,elmin,elmax,
     .                        gbmin,gbmax,glmin,glmax,
     .                        jmin,jmax,hmin,hmax,kmin,kmax))
                  if (usesky_x10(iiu,jju).eq.1) then
                     do iii = iiu-1, iiu+1
                     do jjj = jju-1, jju+1
                        iiiu = iii
                        jjju = jjj
                        if (iiu.lt.0001) iiiu = iiiu+3600
                        if (iiu.gt.3600) iiiu = iiiu-3600
                        if (jju.lt.0001) jjju = 0001
                        if (jju.gt.3600) jjju = 1800
                        usesky_c10(iiiu,jjju) = 1
                        if (iiiu.lt.IIMIN) IIMIN = iiiu
                        if (iiiu.gt.IIMAX) IIMAX = iiiu
                        if (jjju.lt.JJMIN) JJMIN = jjju
                        if (jjju.gt.JJMAX) JJMAX = jjju
                        enddo
                        enddo
                     endif
                  enddo
                enddo
             endif
          enddo
          enddo
c     print*,'close(78)'
 
c     print*,'IIMIN/MAX: ',IIMIN,IIMAX
c     print*,'JJMIN/MAX: ',JJMIN,JJMAX
 
c     call writfits_b1('usesky_x10.fits',usesky_x10,3600,1800)
c     call writfits_b1('usesky_c10.fits',usesky_c10,3600,1800)
 
      dd_lcky = 999.999
      ra_lcky = 0.00
      de_lcky = 0.00
      mj_lcky = 99.99
      mh_lcky = 99.99
      mk_lcky = 99.99
      aa_lcky = '???'
      dx_lcky = 99.99
      dy_lcky = 99.99
      Us = 0
      do jj = JJMIN, JJMAX
         IIMINu = IIMAX
         IIMAXu = IIMIN
         do ii = IIMIN, IIMAX
            if (usesky_c10(ii,jj).eq.1) then
               if (ii.lt.IIMINu) IIMINu = ii
               if (ii.gt.IIMAXu) IIMAXu = ii
               endif
            enddo
         nn1 = max(NMAP_IJ(IIMINu  ,jj),rec0+1)
         nn2 = max(NMAP_IJ(IIMAXu+1,jj),rec0+1)
         rec1 = nn1/277 + 1
         rec2 = nn2/277 + 1
        !print*,jj,IIMINU,IIMAXu,nn1,nn2,rec1,rec2,Us
         do N = rec1, rec2
            read(12,rec=N) buff13_277
            do NN = 1, 277
               call buff2rdm(buff13_277(1,NN),ira,ide,
     .                                        imj,imh,imk,aaa)
               ra = ira/60./60./100.
               de = -90 + ide/60./60./100.
               jmag = imj/100.
               hmag = imh/100.
               kmag = imk/100.
               U = use(ra,de,jmag,hmag,kmag,
     .                 rcen,dcen,jcen,hcen,kcen,
     .                      drmin,drmax,
     .                      djmin,djmax,dhmin,dhmax,dkmin,dkmax,
     .                      dxmax,dymax,
     .                 rmin,rmax,dmin,dmax,
     .                 ebmin,ebmax,elmin,elmax,
     .                 gbmin,gbmax,glmin,glmax,
     .                 jmin,jmax,hmin,hmax,kmin,kmax)
               if (U.eq.1) then
                  Us = Us + 1
                  Os = Us
                  ra_o(Os) = ra
                  de_o(Os) = de
                  mj_o(Os) = jmag
                  mh_o(Os) = hmag
                  mk_o(Os) = kmag
                  aa_o(Os) = aaa
                  endif
               enddo!NN
            enddo!N
         enddo!jj
 
 
 
 
      if (LIST) goto 137
      close(12)
 
      return
 
 
 
 777  continue
      print*,' '
      print*,'END OF INPUT FILE...'
      print*,' '
 
      stop
      end
 
 
 
 
c--------------------------------------------------------------------
c
c
      byte function use(r,d,mj,mh,mk,
     .                  rcen,dcen,jcen,hcen,kcen,
     .                       drmin,drmax,
     .                       djmin,djmax,dhmin,dhmax,dkmin,dkmax,
     .                       dxmax,dymax,
     .                  rmin,rmax,dmin,dmax,
     .                  ebmin,ebmax,elmin,elmax,
     .                  gbmin,gbmax,glmin,glmax,
     .                  jmin,jmax,hmin,hmax,kmin,kmax)
      implicit none
 
      real*8 r,d
      real   mj,mh,mk
 
      real*8 rcen, dcen
      real   jcen, hcen, kcen
      real   drmin, drmax
      real   djmin, djmax, dhmin, dhmax, dkmin, dkmax
      real   dxmax, dymax
      real*8  rmin,  rmax,  dmin,  dmax
      real*8 ebmin, ebmax, elmin, elmax
      real*8 gbmin, gbmax, glmin, glmax
      real   jmin, jmax
      real   hmin, hmax
      real   kmin, kmax
 
      real*8 gl, gb
      real*8 el, eb
 
      real*8 rd2x, rd2y
      real*8 dx, dy, dd
 
      if (rmin.gt.0) then
         use = -1
         if (r.lt.rmin) return
         use = -2
         if (r.gt.rmax) return
         use = -3
         if (d.lt.dmin) return
         use = -4
         if (d.gt.dmax) return
         endif
 
      if (jmin.gt.-9.99) then
         use = -13
         if (mj.lt.jmin) return
         use = -14
         if (mj.gt.jmax) return
         endif
 
      if (hmin.gt.-9.99) then
         use = -15
         if (mh.lt.hmin) return
         use = -16
         if (mh.gt.hmax) return
         endif
 
      if (kmin.gt.-9.99) then
         use = -17
         if (mk.lt.kmin) return
         use = -18
         if (mk.gt.kmax) return
         endif
 
      if (rcen.ge.0.00) then
         dx = rd2x(r,d,rcen,dcen)
         dy = rd2y(r,d,rcen,dcen)
         dd = sqrt(dx**2+dy**2)
         use = -19
         if (dd.lt.drmin) return
         use = -20
         if (dd.gt.drmax) return
         use = -21
         if (mj-jcen.gt.djmax) return
         use = -22
         if (mj-jcen.lt.djmin) return
         use = -23
         if (mh-hcen.gt.dhmax) return
         use = -24
         if (mh-hcen.lt.dhmin) return
         use = -25
         if (mk-kcen.gt.dkmax) return
         use = -26
         if (mk-kcen.lt.dkmin) return
         use = -27
         if (abs(dx).gt.dxmax) return
         use = -28
         if (abs(dy).gt.dymax) return
         endif
 
      use = 1
 
      return
      end
 
 
 
 
      subroutine buff2rdmX(buff_12,ra,de,jmag,hmag,kmag,QUAL)
      implicit none
 
      byte        buff_12(12)
      real*8      ra, de
      real        jmag, hmag, kmag
      character*3 QUAL
 
      integer     b
 
      integer ira, irao
      integer ide, ideo
      integer*2 imj, jqual
      integer*2 imh, hqual
      integer*2 imk, kqual
 
      byte        buffb_i4(4)
      integer*4   buffi_i4
      equivalence (buffb_i4,buffi_i4)
 
      byte        buffb_i2(2)
      integer*2   buffi_i2
      equivalence (buffb_i2,buffi_i2)
 
      byte buffb_ra(3)
      byte buffb_de(3)
      byte buffb_mj(2)
      byte buffb_mh(2)
      byte buffb_mk(2)
 
 
      irao = 0
      ideo = 0
 
      buffb_ra(01) = buff_12(01)
      buffb_ra(02) = buff_12(02)
      buffb_ra(03) = buff_12(03)
      buffb_de(01) = buff_12(04)
      buffb_de(02) = buff_12(05)
      buffb_de(03) = buff_12(06)
      buffb_mj(01) = buff_12(07)
      buffb_mj(02) = buff_12(08)
      buffb_mh(01) = buff_12(09)
      buffb_mh(02) = buff_12(10)
      buffb_mk(01) = buff_12(11)
      buffb_mk(02) = buff_12(12)
 
cc---------------------------------
      buffb_i4(1) = buffb_ra(1)
      buffb_i4(2) = buffb_ra(2)
      buffb_i4(3) = buffb_ra(3)
      buffb_i4(4) = 0
      ira = buffi_i4
 
cc---------------------------------
      buffb_i4(1) = buffb_de(1)
      buffb_i4(2) = buffb_de(2)
      buffb_i4(3) = buffb_de(3)
      buffb_i4(4) = 0
      ide = buffi_i4
 
cc---------------------------------
      buffb_i2(1) = buffb_mj(1)
      buffb_i2(2) = buffb_mj(2)
      imj   = buffi_i2/8
 
      jqual = buffi_i2 - buffi_i2/8*8
      if (jqual.eq.0) QUAL(1:1) = 'X'
      if (jqual.eq.1) QUAL(1:1) = 'U'
      if (jqual.eq.2) QUAL(1:1) = 'F'
      if (jqual.eq.3) QUAL(1:1) = 'E'
      if (jqual.eq.4) QUAL(1:1) = 'A'
      if (jqual.eq.5) QUAL(1:1) = 'B'
      if (jqual.eq.6) QUAL(1:1) = 'C'
      if (jqual.eq.7) QUAL(1:1) = 'D'
 
 
cc---------------------------------
      buffb_i2(1) = buffb_mh(1)
      buffb_i2(2) = buffb_mh(2)
      imh = buffi_i2/8
 
      hqual = buffi_i2 - buffi_i2/8*8
      if (hqual.eq.0) QUAL(2:2) = 'X'
      if (hqual.eq.1) QUAL(2:2) = 'U'
      if (hqual.eq.2) QUAL(2:2) = 'F'
      if (hqual.eq.3) QUAL(2:2) = 'E'
      if (hqual.eq.4) QUAL(2:2) = 'A'
      if (hqual.eq.5) QUAL(2:2) = 'B'
      if (hqual.eq.6) QUAL(2:2) = 'C'
      if (hqual.eq.7) QUAL(2:2) = 'D'
 
cc---------------------------------
      buffb_i2(1) = buffb_mk(1)
      buffb_i2(2) = buffb_mk(2)
      imk = buffi_i2/8
 
      kqual = buffi_i2 - buffi_i2/8*8
      if (kqual.eq.0) QUAL(3:3) = 'X'
      if (kqual.eq.1) QUAL(3:3) = 'U'
      if (kqual.eq.2) QUAL(3:3) = 'F'
      if (kqual.eq.3) QUAL(3:3) = 'E'
      if (kqual.eq.4) QUAL(3:3) = 'A'
      if (kqual.eq.5) QUAL(3:3) = 'B'
      if (kqual.eq.6) QUAL(3:3) = 'C'
      if (kqual.eq.7) QUAL(3:3) = 'D'
 
      ra   = ira/10000.0
      de   = ide/10000.0 - 90.00
      jmag = imj/100.
      hmag = imh/100.
      kmag = imk/100.
 
      end
 
 
 
 
c---------------------------------------------------
c
      integer function ubyte(b)
      implicit none
 
      byte b
      ubyte = b
      if (ubyte.lt.0) ubyte = ubyte + 256
 
      return
      end
 
 
c---------------------------------------------------
c
c designed to compress better...
c
c
      subroutine rdm2buff(ra,de,jmag,hmag,kmag,QUAL,buff13)
      implicit none
 
      real*8      ra, de
      real        jmag, hmag, kmag
      character*3 QUAL
      byte      buff13(13)
 
      integer     b
 
      integer*4 ira, ira1, ira2, iras
      integer*4 ide, ide1, ide2, ides
      integer*2 imj, jqual
      integer*2 imh, hqual
      integer*2 imk, kqual
 
      integer*4 irax
      integer*4 idex
 
      integer*2 jmags, jmagx
      integer*2 hmags, hmagx
      integer*2 kmags, kmagx
 
      byte b32_rai(32)
      byte b32_dei(32)
      byte b16_mji(16)
      byte b16_mhi(16)
      byte b16_mki(16)
 
      integer*4 ira0
      integer*4 ide0
      integer*2 ijm0
      integer*2 ihm0
      integer*2 ikm0
 
      ira = ((ra   )*60*60*100) + 0.5 ! max: 129,600,000 :  2**27 = 134,217,728  ! 8+8+8+3 bytes
      ide = ((de+90)*60*60*100) + 0.5 ! max:  64,800,000 :  2**26 =  67,108,864  ! 8+8+8+2 bytes
 
 
      ira1 =  ira/36000
      ira2 = (ira-ira/36000*36000)
 
      ide1 =  ide/36000
      ide2 = (ide-ide/36000*36000)
 
      iras = ira1*65536 + ira2
      ides = ide1*65536 + ide2
 
      jmags = abs(jmag*100) + 0.5
      if (jmags.gt. 2000) jmags =  2000
      if (jmag.lt.0) jmags = -jmags
 
      hmags = abs(hmag*100) + 0.5
      if (hmags.gt. 2000) hmags =  2000
      if (hmag.lt.0) hmags = -hmags
 
      kmags = abs(kmag*100) + 0.5
      if (kmags.gt. 2000) kmags =  2000
      if (kmag.lt.0) kmags = -kmags
 
      ira0 = ira
      ide0 = ide
      ijm0 = jmags
      ihm0 = hmags
      ikm0 = kmags
 
      jqual = -1
      if (QUAL(1:1).eq.'X') jqual = 0
      if (QUAL(1:1).eq.'U') jqual = 1
      if (QUAL(1:1).eq.'F') jqual = 2
      if (QUAL(1:1).eq.'E') jqual = 3
      if (QUAL(1:1).eq.'A') jqual = 4
      if (QUAL(1:1).eq.'B') jqual = 5
      if (QUAL(1:1).eq.'C') jqual = 6
      if (QUAL(1:1).eq.'D') jqual = 7
 
 
      hqual = -1
      if (QUAL(2:2).eq.'X') hqual = 0
      if (QUAL(2:2).eq.'U') hqual = 1
      if (QUAL(2:2).eq.'F') hqual = 2
      if (QUAL(2:2).eq.'E') hqual = 3
      if (QUAL(2:2).eq.'A') hqual = 4
      if (QUAL(2:2).eq.'B') hqual = 5
      if (QUAL(2:2).eq.'C') hqual = 6
      if (QUAL(2:2).eq.'D') hqual = 7
 
      kqual = -1
      if (QUAL(3:3).eq.'X') kqual = 0
      if (QUAL(3:3).eq.'U') kqual = 1
      if (QUAL(3:3).eq.'F') kqual = 2
      if (QUAL(3:3).eq.'E') kqual = 3
      if (QUAL(3:3).eq.'A') kqual = 4
      if (QUAL(3:3).eq.'B') kqual = 5
      if (QUAL(3:3).eq.'C') kqual = 6
      if (QUAL(3:3).eq.'D') kqual = 7
 
      if (jqual.eq.-1.or.hqual.eq.-1.or.kqual.eq.-1) then
         write(*,*) 'jqual: ',jqual,QUAL(1:1)
         write(*,*) 'hqual: ',hqual,QUAL(2:2)
         write(*,*) 'kqual: ',kqual,QUAL(3:3)
         stop
         endif
 
      imj = abs(jmags)*8 + jqual
      imh = abs(hmags)*8 + hqual
      imk = abs(kmags)*8 + kqual
 
      if (jmags.lt.0) imj = -imj
      if (hmags.lt.0) imh = -imh
      if (kmags.lt.0) imk = -imk
 
 
c
c-------------------------------------------------------------
c
      do b = 1, 32
         b32_dei(b) = 0
         b32_rai(b) = 0
         enddo
      do b = 1, 16
         b16_mji(b) = 0
         b16_mhi(b) = 0
         b16_mki(b) = 0
         enddo
 
      call i4_to_b32(iras,b32_rai)
      call i4_to_b32(ides,b32_dei)
 
 
      call i2_to_b16(imj,b16_mji)
      call i2_to_b16(imh,b16_mhi)
      call i2_to_b16(imk,b16_mki)
 
c
c-------------------------------------------------------------
c
      call bbbbbbbb_to_b1(buff13(01),
     .        b32_dei(05),b32_dei(06),b32_dei(07),b32_dei(08),     ! 1st nybble in de_a
     .        b32_dei(09),b32_dei(10),b32_dei(11),b32_dei(12))     ! 2nd nybble in de_a
 
      call bbbbbbbb_to_b1(buff13(02),
     .        b32_dei(13),b32_dei(14),b32_dei(15),b32_dei(16),     ! 3rd nybble in de_a
     .        b32_rai(05),b32_rai(06),b32_rai(07),b32_rai(08))     ! 1st nybble in ra_a
 
      call bbbbbbbb_to_b1(buff13(03),
     .        b32_rai(09),b32_rai(10),b32_rai(11),b32_rai(12),     ! 2nd nybble in ra_a
     .        b32_rai(13),b32_rai(14),b32_rai(15),b32_rai(16))     ! 3rd nybble in ra_a
 
      call bbbbbbbb_to_b1(buff13(04),
     .        b16_mji(01), b16_mhi(01), b16_mki(01),               ! 1st byte in mjhk
     .        b16_mji(02), b16_mhi(02), b16_mki(02),               ! 2nd byte in mjhk
     .        b16_mji(03), b16_mhi(03))                            ! 3rd byte in mjhk (2/3)
 
      call bbbbbbbb_to_b1(buff13(05),
     .                                  b16_mki(03),               ! 3rd byte in mjhk (1/3)
     .        b16_mji(04), b16_mhi(04), b16_mki(04),               ! 4th byte in mjhk
     .        b16_mji(05), b16_mhi(05), b16_mki(05),               ! 5th byte in mjhk
     .        b16_mji(06))                                         ! 6th byte in mjhk (1/3)
 
      call bbbbbbbb_to_b1(buff13(06),
     .                     b16_mhi(06), b16_mki(06),               ! 6th byte in mjhk (2/3)
     .        b16_mji(07), b16_mhi(07), b16_mki(07),               ! 7th byte in mjhk
     .        b16_mji(08), b16_mhi(08), b16_mki(08))               ! 8th byte in mjhk
 
      call bbbbbbbb_to_b1(buff13(07),
     .        b16_mji(09), b16_mji(10), b16_mji(11), b16_mji(12),  ! last byte in mj
     .        b16_mji(13), b16_mji(14), b16_mji(15), b16_mji(16))
 
      call bbbbbbbb_to_b1(buff13(08),
     .        b16_mhi(09), b16_mhi(10), b16_mhi(11), b16_mhi(12),  ! last byte in mh
     .        b16_mhi(13), b16_mhi(14), b16_mhi(15), b16_mhi(16))
 
      call bbbbbbbb_to_b1(buff13(09),
     .        b16_mki(09), b16_mki(10), b16_mki(11), b16_mki(12),  ! last byte in mk
     .        b16_mki(13), b16_mki(14), b16_mki(15), b16_mki(16))
 
      call bbbbbbbb_to_b1(buff13(10),
     .        b32_rai(17), b32_rai(18), b32_rai(19), b32_rai(20),
     .        b32_rai(21), b32_rai(22), b32_rai(23), b32_rai(24))
 
      call bbbbbbbb_to_b1(buff13(11),
     .        b32_rai(25), b32_rai(26), b32_rai(27), b32_rai(28),
     .        b32_rai(29), b32_rai(30), b32_rai(31), b32_rai(32))
 
      call bbbbbbbb_to_b1(buff13(12),
     .        b32_dei(17), b32_dei(18), b32_dei(19), b32_dei(20),
     .        b32_dei(21), b32_dei(22), b32_dei(23), b32_dei(24))
 
      call bbbbbbbb_to_b1(buff13(13),
     .        b32_dei(25), b32_dei(26), b32_dei(27), b32_dei(28),
     .        b32_dei(29), b32_dei(30), b32_dei(31), b32_dei(32))
 
      return
      end
 
 
 
c---------------------------------------------------
c
c designed to compress better...
c
c
      subroutine buff2rdm(buff13,ira,ide,imj,imh,imk,aaa)
     .
      implicit none
 
      byte      buff13(13)
      integer*4   ira, ide
      integer*2   imj, imh, imk
      character*3 aaa
 
      integer*4 ira1, ira2, irau
      integer*4 ide1, ide2, ideu
      integer*2 imju, jqual
      integer*2 imhu, hqual
      integer*2 imku, kqual
 
      byte b
 
      integer*4 irax
      integer*4 idex
 
      integer*2 jmagu, jmagx
      integer*2 hmagu, hmagx
      integer*2 kmagu, kmagx
 
      byte b32_rai(32)
      byte b32_dei(32)
      byte b16_mji(16)
      byte b16_mhi(16)
      byte b16_mki(16)
 
      real*8 ra, de
      real*8 jmag, hmag, kmag
 
c---------------------------------------------------------------------------
c
c now, back out what we put in, using only the buff information
c
      do b = 1, 32
         b32_dei(b) = 0
         b32_rai(b) = 0
         enddo
      do b = 1, 16
         b16_mji(b) = 0
         b16_mhi(b) = 0
         b16_mki(b) = 0
         enddo
      ira1 = 0
      ira2 = 0
      ide1 = 0
      ide2 = 0
      imj  = 0
      imh  = 0
      imk  = 0
      jmagu = 0
      hmagu = 0
      kmagu = 0
      AAA = '   '
 
 
c---------------------------------------------------------------------------
c
      call b1_to_bbbbbbbb(buff13(01),
     .        b32_dei(05),b32_dei(06),b32_dei(07),b32_dei(08),     ! 1st nybble in de_a
     .        b32_dei(09),b32_dei(10),b32_dei(11),b32_dei(12))     ! 2nd nybble in de_a
 
      call b1_to_bbbbbbbb(buff13(02),
     .        b32_dei(13),b32_dei(14),b32_dei(15),b32_dei(16),     ! 3rd nybble in de_a
     .        b32_rai(05),b32_rai(06),b32_rai(07),b32_rai(08))     ! 1st nybble in ra_a
 
      call b1_to_bbbbbbbb(buff13(03),
     .        b32_rai(09),b32_rai(10),b32_rai(11),b32_rai(12),     ! 2nd nybble in ra_a
     .        b32_rai(13),b32_rai(14),b32_rai(15),b32_rai(16))     ! 3rd nybble in ra_a
 
      call b1_to_bbbbbbbb(buff13(04),
     .        b16_mji(01), b16_mhi(01), b16_mki(01),               ! 1st byte in mjhk
     .        b16_mji(02), b16_mhi(02), b16_mki(02),               ! 2nd byte in mjhk
     .        b16_mji(03), b16_mhi(03))                            ! 3rd byte in mjhk (2/3)
 
      call b1_to_bbbbbbbb(buff13(05),
     .                                  b16_mki(03),               ! 3rd byte in mjhk (1/3)
     .        b16_mji(04), b16_mhi(04), b16_mki(04),               ! 4th byte in mjhk
     .        b16_mji(05), b16_mhi(05), b16_mki(05),               ! 5th byte in mjhk
     .        b16_mji(06))                                         ! 6th byte in mjhk (1/3)
 
      call b1_to_bbbbbbbb(buff13(06),
     .                     b16_mhi(06), b16_mki(06),               ! 6th byte in mjhk (2/3)
     .        b16_mji(07), b16_mhi(07), b16_mki(07),               ! 7th byte in mjhk
     .        b16_mji(08), b16_mhi(08), b16_mki(08))               ! 8th byte in mjhk
 
      call b1_to_bbbbbbbb(buff13(07),
     .        b16_mji(09), b16_mji(10), b16_mji(11), b16_mji(12),  ! last byte in mj
     .        b16_mji(13), b16_mji(14), b16_mji(15), b16_mji(16))
 
      call b1_to_bbbbbbbb(buff13(08),
     .        b16_mhi(09), b16_mhi(10), b16_mhi(11), b16_mhi(12),  ! last byte in mh
     .        b16_mhi(13), b16_mhi(14), b16_mhi(15), b16_mhi(16))
 
      call b1_to_bbbbbbbb(buff13(09),
     .        b16_mki(09), b16_mki(10), b16_mki(11), b16_mki(12),  ! last byte in mk
     .        b16_mki(13), b16_mki(14), b16_mki(15), b16_mki(16))
 
      call b1_to_bbbbbbbb(buff13(10),
     .        b32_rai(17), b32_rai(18), b32_rai(19), b32_rai(20),
     .        b32_rai(21), b32_rai(22), b32_rai(23), b32_rai(24))
 
      call b1_to_bbbbbbbb(buff13(11),
     .        b32_rai(25), b32_rai(26), b32_rai(27), b32_rai(28),
     .        b32_rai(29), b32_rai(30), b32_rai(31), b32_rai(32))
 
      call b1_to_bbbbbbbb(buff13(12),
     .        b32_dei(17), b32_dei(18), b32_dei(19), b32_dei(20),
     .        b32_dei(21), b32_dei(22), b32_dei(23), b32_dei(24))
 
      call b1_to_bbbbbbbb(buff13(13),
     .        b32_dei(25), b32_dei(26), b32_dei(27), b32_dei(28),
     .        b32_dei(29), b32_dei(30), b32_dei(31), b32_dei(32))
 
      call b32_to_i4(b32_rai,irau)
      call b32_to_i4(b32_dei,ideu)
 
      call b16_to_i2(b16_mji,imj)
      call b16_to_i2(b16_mhi,imh)
      call b16_to_i2(b16_mki,imk)
 
      ira1 = irau/65536
      ira2 = irau-irau/65536*65536
 
      ide1 = ideu/65536
      ide2 = ideu-ideu/65536*65536
 
      ira  = ira1*36000 + ira2
      ide  = ide1*36000 + ide2
 
      jqual = abs(imj)-abs(imj)/8*8
      hqual = abs(imh)-abs(imh)/8*8
      kqual = abs(imk)-abs(imk)/8*8
 
      jmagu = abs(imj)/8
      hmagu = abs(imh)/8
      kmagu = abs(imk)/8
      if (imj.lt.0) jmagu = -jmagu
      if (imh.lt.0) hmagu = -hmagu
      if (imk.lt.0) kmagu = -kmagu
 
      if (jqual.eq.0) AAA(1:1) = 'X'
      if (jqual.eq.1) AAA(1:1) = 'U'
      if (jqual.eq.2) AAA(1:1) = 'F'
      if (jqual.eq.3) AAA(1:1) = 'E'
      if (jqual.eq.4) AAA(1:1) = 'A'
      if (jqual.eq.5) AAA(1:1) = 'B'
      if (jqual.eq.6) AAA(1:1) = 'C'
      if (jqual.eq.7) AAA(1:1) = 'D'
 
      if (hqual.eq.0) AAA(2:2) = 'X'
      if (hqual.eq.1) AAA(2:2) = 'U'
      if (hqual.eq.2) AAA(2:2) = 'F'
      if (hqual.eq.3) AAA(2:2) = 'E'
      if (hqual.eq.4) AAA(2:2) = 'A'
      if (hqual.eq.5) AAA(2:2) = 'B'
      if (hqual.eq.6) AAA(2:2) = 'C'
      if (hqual.eq.7) AAA(2:2) = 'D'
 
      if (kqual.eq.0) AAA(3:3) = 'X'
      if (kqual.eq.1) AAA(3:3) = 'U'
      if (kqual.eq.2) AAA(3:3) = 'F'
      if (kqual.eq.3) AAA(3:3) = 'E'
      if (kqual.eq.4) AAA(3:3) = 'A'
      if (kqual.eq.5) AAA(3:3) = 'B'
      if (kqual.eq.6) AAA(3:3) = 'C'
      if (kqual.eq.7) AAA(3:3) = 'D'
 
      imj = jmagu
      imh = hmagu
      imk = kmagu
 
      return
      end
 
 
c----------------------------------------------
c
      subroutine i4_to_b32(i4,b32)
      implicit none
 
      integer*4 i4
      byte    b32(32)
 
      integer*4 i4u
 
      i4u = i4
 
      b32(01) = 0
      if (i4u.lt.0) then
         b32(01) = 0
         i4u = -i4u
         endif
 
      b32(02) = i4u/2**30
      b32(03) = (i4u-i4u/2**30*2**30)/2**29
      b32(04) = (i4u-i4u/2**29*2**29)/2**28
      b32(05) = (i4u-i4u/2**28*2**28)/2**27
      b32(06) = (i4u-i4u/2**27*2**27)/2**26
      b32(07) = (i4u-i4u/2**26*2**26)/2**25
      b32(08) = (i4u-i4u/2**25*2**25)/2**24
      b32(09) = (i4u-i4u/2**24*2**24)/2**23
      b32(10) = (i4u-i4u/2**23*2**23)/2**22
      b32(11) = (i4u-i4u/2**22*2**22)/2**21
      b32(12) = (i4u-i4u/2**21*2**21)/2**20
      b32(13) = (i4u-i4u/2**20*2**20)/2**19
      b32(14) = (i4u-i4u/2**19*2**19)/2**18
      b32(15) = (i4u-i4u/2**18*2**18)/2**17
      b32(16) = (i4u-i4u/2**17*2**17)/2**16
      b32(17) = (i4u-i4u/2**16*2**16)/2**15
      b32(18) = (i4u-i4u/2**15*2**15)/2**14
      b32(19) = (i4u-i4u/2**14*2**14)/2**13
      b32(20) = (i4u-i4u/2**13*2**13)/2**12
      b32(21) = (i4u-i4u/2**12*2**12)/2**11
      b32(22) = (i4u-i4u/2**11*2**11)/2**10
      b32(23) = (i4u-i4u/2**10*2**10)/2**09
      b32(24) = (i4u-i4u/2**09*2**09)/2**08
      b32(25) = (i4u-i4u/2**08*2**08)/2**07
      b32(26) = (i4u-i4u/2**07*2**07)/2**06
      b32(27) = (i4u-i4u/2**06*2**06)/2**05
      b32(28) = (i4u-i4u/2**05*2**05)/2**04
      b32(29) = (i4u-i4u/2**04*2**04)/2**03
      b32(30) = (i4u-i4u/2**03*2**03)/2**02
      b32(31) = (i4u-i4u/2**02*2**02)/2**01
      b32(32) = (i4u-i4u/2*2)
 
      return
      end
 
 
 
 
c----------------------------------------------
c
      subroutine i2_to_b16(i2,b16)
      implicit none
 
      integer*2 i2
      byte    b16(16)
 
      integer*2  i2u
      integer b
 
 
      i2u = i2
      b16(01) = 0
      if (i2u.lt.0) then
         b16(01) = 1
         i2u = -i2u
         endif
 
      b16(02) = (i2u-i2u/2**15*2**15)/2**14
      b16(03) = (i2u-i2u/2**14*2**14)/2**13
      b16(04) = (i2u-i2u/2**13*2**13)/2**12
      b16(05) = (i2u-i2u/2**12*2**12)/2**11
      b16(06) = (i2u-i2u/2**11*2**11)/2**10
      b16(07) = (i2u-i2u/2**10*2**10)/2**09
      b16(08) = (i2u-i2u/2**09*2**09)/2**08
      b16(09) = (i2u-i2u/2**08*2**08)/2**07
      b16(10) = (i2u-i2u/2**07*2**07)/2**06
      b16(11) = (i2u-i2u/2**06*2**06)/2**05
      b16(12) = (i2u-i2u/2**05*2**05)/2**04
      b16(13) = (i2u-i2u/2**04*2**04)/2**03
      b16(14) = (i2u-i2u/2**03*2**03)/2**02
      b16(15) = (i2u-i2u/2**02*2**02)/2**01
      b16(16) = (i2u-i2u/2*2)
 
      return
      end
 
 
c----------------------------------------------
c
      subroutine b1_to_b8(b1,b8)
      implicit none
 
      byte    b1
      byte    b8(8)
 
      integer   b1u
 
      b1u = b1
 
      if (b1u.lt.0) b1u = b1u + 256
 
      b8(01) = (b1u-b1u/2**08*2**08)/2**07
      b8(02) = (b1u-b1u/2**07*2**07)/2**06
      b8(03) = (b1u-b1u/2**06*2**06)/2**05
      b8(04) = (b1u-b1u/2**05*2**05)/2**04
      b8(05) = (b1u-b1u/2**04*2**04)/2**03
      b8(06) = (b1u-b1u/2**03*2**03)/2**02
      b8(07) = (b1u-b1u/2**02*2**02)/2**01
      b8(08) = (b1u-b1u/2**01*2**01)/2**00
 
      return
      end
 
 
c----------------------------------------------
c
      subroutine b8_to_b1(b8,b1)
      implicit none
 
      byte    b8(8)
      byte    b1
 
 
      b1 = b8(8)*2**0
     .   + b8(7)*2**1
     .   + b8(6)*2**2
     .   + b8(5)*2**3
     .   + b8(4)*2**4
     .   + b8(3)*2**5
     .   + b8(2)*2**6
 
      if (b8(1).eq.1) b1 = -b1
 
      return
      end
 
 
c----------------------------------------------
c
      subroutine b16_to_i2(b16,i2)
      implicit none
 
      byte      b16(16)
      integer*2 i2
 
 
      i2 = b16(16)*2**00
     .   + b16(15)*2**01
     .   + b16(14)*2**02
     .   + b16(13)*2**03
     .   + b16(12)*2**04
     .   + b16(11)*2**05
     .   + b16(10)*2**06
     .   + b16(09)*2**07
     .   + b16(08)*2**08
     .   + b16(07)*2**09
     .   + b16(06)*2**10
     .   + b16(05)*2**11
     .   + b16(04)*2**12
     .   + b16(03)*2**13
     .   + b16(02)*2**14
 
      if (b16(01).eq.1) i2 = -i2
 
      return
      end
 
 
c----------------------------------------------
c
      subroutine b32_to_i4(b32,i4)
      implicit none
 
      byte      b32(32)
      integer*4 i4
 
      i4 = b32(32)*2**00
     .   + b32(31)*2**01
     .   + b32(30)*2**02
     .   + b32(29)*2**03
     .   + b32(28)*2**04
     .   + b32(27)*2**05
     .   + b32(26)*2**06
     .   + b32(25)*2**07
     .   + b32(24)*2**08
     .   + b32(23)*2**09
     .   + b32(22)*2**10
     .   + b32(21)*2**11
     .   + b32(20)*2**12
     .   + b32(19)*2**13
     .   + b32(18)*2**14
     .   + b32(17)*2**15
     .   + b32(16)*2**16
     .   + b32(15)*2**17
     .   + b32(14)*2**18
     .   + b32(13)*2**19
     .   + b32(12)*2**20
     .   + b32(11)*2**21
     .   + b32(10)*2**22
     .   + b32(09)*2**23
     .   + b32(08)*2**24
     .   + b32(07)*2**25
     .   + b32(06)*2**26
     .   + b32(05)*2**27
     .   + b32(04)*2**28
     .   + b32(03)*2**29
     .   + b32(02)*2**30
      if (b32(01).eq.1) i4 = -i4
 
      return
      end
 
 
c----------------------------------------------------------
c
      subroutine bbbbbbbb_to_b1(b1,ba,bb,bc,bd,be,bf,bg,bh)
      implicit none
 
      byte b1
      byte ba,bb,bc,bd
      byte be,bf,bg,bh
 
      integer b1u
 
      byte bax,bbx,bcx,bdx
      byte bex,bfx,bgx,bhx
 
      b1u = 0
      b1u = b1u
     .    + bh
     .    + bg*002
     .    + bf*004
     .    + be*008
     .    + bd*016
     .    + bc*032
     .    + bb*064
     .    + ba*128
 
      b1 = b1u
 
      call b1_to_bbbbbbbb(b1,bax,bbx,bcx,bdx,bex,bfx,bgx,bhx)
 
      if (ba.ne.bax.or.
     .    bb.ne.bbx.or.
     .    bc.ne.bcx.or.
     .    bd.ne.bdx.or.
     .    be.ne.bex.or.
     .    bf.ne.bfx.or.
     .    bg.ne.bgx.or.
     .    bh.ne.bhx) then
             print*,'b1 : ',b1
             print*,'b1u: ',b1u
             print*,'  a: ',ba,bax
             print*,'  b: ',bb,bbx
             print*,'  c: ',bc,bcx
             print*,'  d: ',bd,bdx
             print*,'  e: ',be,bex
             print*,'  f: ',bf,bfx
             print*,'  g: ',bg,bgx
             print*,'  h: ',bh,bhx
             stop
             endif
 
      return
      end
 
c----------------------------------------------------------
c
      subroutine b1_to_bbbbbbbb(b1,ba,bb,bc,bd,be,bf,bg,bh)
      implicit none
 
      byte b1
      byte ba,bb,bc,bd
      byte be,bf,bg,bh
 
      integer*2 b1u
 
      b1u = b1
      if (b1u.lt.0) b1u = b1u + 256
 
      ba = (b1u-b1u/2**08*2**08)/2**07
      bb = (b1u-b1u/2**07*2**07)/2**06
      bc = (b1u-b1u/2**06*2**06)/2**05
      bd = (b1u-b1u/2**05*2**05)/2**04
      be = (b1u-b1u/2**04*2**04)/2**03
      bf = (b1u-b1u/2**03*2**03)/2**02
      bg = (b1u-b1u/2**02*2**02)/2**01
      bh = (b1u-b1u/2**01*2**01)/2**00
 
      return
      end
 
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/WCS/sub_find_ifilt_acswfc.f"
c****                                         
c*********************************************
                                              
 
 
      subroutine sub_find_ifilt_acswfc(FILENAME,IFILT,FILTNAME)
      implicit none
 
      character*80 FILENAME
      integer IFILT
      character*05 FILTNAME
 
      character*20 STREAM
 
      IFILT = 0
      FILTNAME = 'NONE'
 
      call query_hdre(FILENAME, 'FILTER1  ',STREAM,-1)
      if (STREAM(2:2).eq.'C')
     .call query_hdre(FILENAME, 'FILTER2  ',STREAM,-1)
      if (STREAM(2:6).eq.'F435W') IFILT = 01
      if (STREAM(2:6).eq.'F475W') IFILT = 02
      if (STREAM(2:6).eq.'F555W') IFILT = 03
      if (STREAM(2:6).eq.'F606W') IFILT = 04
      if (STREAM(2:6).eq.'F814W') IFILT = 05
      if (STREAM(2:6).eq.'F850W') IFILT = 06
      if (STREAM(2:6).eq.'F625W') IFILT = 07
      if (STREAM(2:6).eq.'F658W') IFILT = 08
      if (STREAM(2:6).eq.'F775W') IFILT = 09
      if (STREAM(2:6).eq.'F660N') IFILT = 10
      if (STREAM(2:6).eq.'F502N') IFILT = 11
      if (STREAM(2:6).eq.'F550N') IFILT = 12
      if (STREAM(2:2).eq.'F') FILTNAME = STREAM(2:6)
 
 
      return
      end
 
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/WCS/sub_find_ifilt_wfc3uv.f"
c****                                         
c*********************************************
                                              
 
 
 
      subroutine sub_find_ifilt_wfc3uv(FILENAME,IFILT,FILTNAME)
      implicit none
 
      character*80 FILENAME
      integer IFILT
      character*05 FILTNAME
 
      character*20 STREAM
 
      IFILT = -1
      FILTNAME = 'NONE'
 
      call query_hdre(FILENAME, 'FILTER   ',STREAM,-1)
      if (STREAM(2:6).eq.'F225W') IFILT = 00
      if (STREAM(2:6).eq.'F275W') IFILT = 01
      if (STREAM(2:6).eq.'F336W') IFILT = 02
      if (STREAM(2:6).eq.'F390W') IFILT = 03
      if (STREAM(2:6).eq.'F300X') IFILT = 03
      if (STREAM(2:6).eq.'F438W') IFILT = 04
      if (STREAM(2:6).eq.'F555W') IFILT = 05
      if (STREAM(2:6).eq.'F606W') IFILT = 06
      if (STREAM(2:6).eq.'F775W') IFILT = 07
      if (STREAM(2:6).eq.'F814W') IFILT = 08
      if (STREAM(2:6).eq.'F850W') IFILT = 09
      if (STREAM(2:2).eq.'F') FILTNAME = STREAM(2:6)
 
      return
      end
 
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/2MASS/find_offset_2MASS.f"
c****                                         
c*********************************************
                                              
 
c-------------------------------------------
c
c
c
c
      subroutine find_offset_2MASS(u1,v1,m1,N1s,
     .                             u2,v2,m2,N2s,
     .                             AG,BG,CG,DG,
     .                             u1o,v1o,u2o,v2o,Lu,Ls,
     .                             dbar,dmax,
     .                             u1mat,v1mat,m1mat,n1mat,
     .                             u2mat,v2mat,m2mat,n2mat,NMATs)
      implicit none
 
      integer N1s
      real*8  u1(N1s), v1(N1s), m1(N1s)
      integer N2s
      real*8  u2(N2s), v2(N2s), m2(N2s)
      real*8  AG,BG,CG,DG
      real*8  u1o,v1o,u2o,v2o
      integer Lu, Ls
      real    dbar, dmax
      real*8  u1mat(_OMAX_), v1mat(_OMAX_), m1mat(_OMAX_)
      integer n1mat(_OMAX_)
      real*8  u2mat(_OMAX_), v2mat(_OMAX_), m2mat(_OMAX_)
      integer n2mat(_OMAX_)
      integer NMATs
 
      integer i,j
      integer hash(1001,1001)
      integer hcsh(1001,1001)
      integer N1, N2
      integer imax, jmax, hmax
      integer ii, jj
      real    dmaxi
 
      integer L
      integer uu_l(_OMAX_)
      real*8  u1_l(_OMAX_), v1_l(_OMAX_), m1_l(_OMAX_)
      real*8  u2_l(_OMAX_), v2_l(_OMAX_), m2_l(_OMAX_)
      integer n1_l(_OMAX_)
      integer n2_l(_OMAX_)
 
      real*8  xbar, xsig
      real*8  ybar, ysig
      real*8  mbar, msig
      real*8  qbar, qsig
 
      real u1min, u1max, u2min, u2max, umin, umax
      real v1min, v1max, v2min, v2max, vmin, vmax
 
      real ri, rj
 
      real*8 GA,GB,GC,GD
      real*4 dbaru,mbaru,msigu,dmaxu
      real   dx, dy, dd, dm
      real*8 u1e, v1e, m1e
      real*8 u2e, v2e, m2e
 
      logical VERBOSE
 
      dmaxi = dmax
 
      VERBOSE = .false.
 
      NMATs = 0
 
      do i = 001, 1001
      do j = 001, 1001
           hash(i,j) = 0
           hcsh(i,j) = 0
           enddo
           enddo
 
      do N1 = 1, N1s
      do N2 = 1, N2s
           i = 501 + (u1(N1)-u2(N2))/3.0 + 0.5
           j = 501 + (v1(N1)-v2(N2))/3.0 + 0.5
           if (i.ge.001.and.i.le.1001.and.
     .         j.ge.001.and.j.le.1001) then
               hash(i,j) = hash(i,j) + 1
               endif
           enddo
           enddo
 
      do i = 0001+3, 1001-3
      do j = 0001+3, 1001-3
           do ii = -3, +3
           do jj = -3, +3
              if (ii**2+jj**2.le.3.5**2) then
                 hcsh(i,j) = hcsh(i,j) + hash(i+ii,j+jj)
                 endif
              enddo
              enddo
           enddo
           enddo
 
      imax = 1
      jmax = 1
      do i = 0001, 1001
      do j = 0001, 1001
           if (hcsh(i,j).gt.hcsh(imax,jmax)) then
              imax = i
              jmax = j
              endif
           enddo
           enddo
 
      if (VERBOSE) then
           print*,'        '
           print*,'  imax: ',imax
           print*,'  jmax: ',jmax
           print*,'        '
           print*,'  hash: ',hash(imax,jmax),imax,jmax
           print*,' '
           write(*,214) (i    ,i=max(1,imax-10),min(1001,imax+10))
           write(*,214) (i-501,i=max(1,imax-10),min(1001,imax+10))
           print*,' '
           do j = min(1001,jmax+10), max(0001,jmax-10),-1
              write(*,213) j,j-501,(hash(i,j),i=max(1,imax-10),
     .                                        min(1001,imax+10))
     .
  213       format(1x,2i6.4,1x,21i5,5x,21i5)
  214       format(1x,2(6x),1x,21i5,5x,21i5)
              enddo
           print*,' '
           print*,' '
           print*,'  hcsh: ',hcsh(imax,jmax),imax,jmax
           print*,' '
           write(*,214) (i-501,i=max(1,imax-10),min(1001,imax+10))
           write(*,214) (i    ,i=max(1,imax-10),min(1001,imax+10))
           print*,' '
           do j = min(1001,jmax+10), max(0001,jmax-10),-1
              write(*,213) j,j-501,(hcsh(i,j),i=max(1,imax-10),
     .                                        min(1001,imax+10))
              enddo
           print*,' '
           write(*,214) (i-501,i=max(1,imax-10),min(1001,imax+10))
           write(*,214) (i    ,i=max(1,imax-10),min(1001,imax+10))
           print*,' '
          !call writfits_i4('hash.fits',hash,1001,1001)
          !call writfits_i4('hcsh.fits',hcsh,1001,1001)
           endif
 
      if ((imax-501)**2+(jmax-501)**2.gt.500**2) then ! failure
           AG = 0.00
           BG = 0.00
           CG = 0.00
           DG = 0.00
           Ls = 0
           Lu = 0
           dbar = 99.9
           dmax = 99.9
           u1o = 0.
           v1o = 0.
           u2o = 0.
           v2o = 0.
           return
           endif
 
      Ls = 0
      do N1 = 1, N1s
      do N2 = 1, N2s
           ri = 501 + (u1(N1)-u2(N2))/3.0
           rj = 501 + (v1(N1)-v2(N2))/3.0
           if ((ri-imax)**2+(rj-jmax)**2.le.7.5**2) then  ! within 15 real pixels
              Ls = Ls + 1
              uu_l(Ls) = 1
              u1_l(Ls) = u1(N1)
              v1_l(Ls) = v1(N1)
              m1_l(Ls) = m1(N1)
              n1_l(Ls) = N1
              u2_l(Ls) = u2(N2)
              v2_l(Ls) = v2(N2)
              m2_l(Ls) = m2(N2)
              n2_l(Ls) = N2
              endif
           enddo
           enddo
 
      call xymxym_purge(u1_l,v1_l,m1_l,
     .                  u2_l,v2_l,m2_l,
     .                  uu_l,Lu,Ls,
     .                  AG,BG,CG,DG,
     .                  u1o,v1o,u2o,v2o,
     .                  mbaru,msigu,
     .                  dbaru,dmaxu, dmaxi, 5.0)
 
 
      do L = 1, Ls
         if (uu_l(L).eq.1) then
             NMATs = NMATs + 1
             u1mat(NMATs) = u1_l(L)
             v1mat(NMATs) = v1_l(L)
             m1mat(NMATs) = m1_l(L)
             n1mat(NMATs) = n1_l(L)
             u2mat(NMATs) = u2_l(L)
             v2mat(NMATs) = v2_l(L)
             m2mat(NMATs) = m2_l(L)
             n2mat(NMATs) = n2_l(L)
             endif
         enddo
 
      dbar = dbaru
      dmax = dmaxu
 
      return
      end
 
 
 
c-----------------------------------------------------------------
c
c this routine will take the x and y arrays
c of peaks (one in each image...) and will find
c the N (1st test: 5) closest and will analyze
c these 5 to produce an average/sigma.  There will
c of course be a bias in this result, but what I
c am doing here is essentially a MODE in 2 dimensions.
c
c I'll carry along the mag info so that I can average
c that for the same reported peaks
c
c
c NTOT : tot number of measts
c NUSE : the number included in the initial "mode"
c        finding (the closest NUSE...)
c NUSED: the actual number included (>=NUSE); I
c        will include those further and further
c        out provided they are consistent with
c        the inner points
c
 
      subroutine barsig2(xl,yl,ml,ql,NTOT,
     .                   xbar,ybar,mbar,qbar,
     .                   xsig,ysig,msig,qsig,
     .                   NMIN,NUSED,wl)
      implicit none
 
      integer NTOT
      real*8 xl(NTOT)
      real*8 yl(NTOT)
      real*8 ml(NTOT)
      real*8 ql(NTOT)
      real*8 xbar, ybar, mbar, qbar
      real*8 xsig, ysig, msig, qsig
      integer NMIN, NUSED
      integer wl(NTOT)
      integer n
      real*8 xsum, ysum, msum, qsum
      real*8 asum, bsum
      real*8 dsig
      real*8  dmax,dmaxx
      integer nmax
 
      real*8 xbarx, ybarx
      real*8 xsigx, ysigx, dsigx
 
      real dx, dy, dd
      integer wsum, wsumx
 
 
      xbar = 0.
      ybar = 0.
      mbar = 0.
      qbar = 0.
      xsig = 1.
      ysig = 1.
      msig = 1.
      qsig = 1.
      NUSED = 0
      if (NTOT.lt.1)    return
      if (NMIN.gt.NTOT) return
 
      if (NTOT.eq.1) then
           xbar = xl(1)
           ybar = yl(1)
           mbar = ml(1)
           qbar = ql(1)
           NUSED = 1
           return
           endif
 
      if (NTOT.eq.2) then
           xbar = (xl(1)+xl(2))/2
           ybar = (yl(1)+yl(2))/2
           mbar = (ml(1)+ml(2))/2
           qbar = (ql(1)+ql(2))/2
           xsig = abs(xl(1)-xl(2))
           ysig = abs(yl(1)-yl(2))
           msig = abs(ml(1)-ml(2))
           qsig = abs(ql(1)-ql(2))
           NUSED = 2
           return
           endif
 
      if (NMIN.gt.NTOT) stop 'BIG PROBLEM'
 
      do n = 1, NTOT
           wl(n) = 1
           enddo
 
      xbar = 000
      ybar = 000
      dsig = 9e9
      dmax = 9e9
  1   continue
 
      xsum = 0.
      ysum = 0.
      msum = 0.
      qsum = 0.
      wsum = 0.
      do n = 1, NTOT
           xsum = xsum + wl(n)*xl(n)
           ysum = ysum + wl(n)*yl(n)
           msum = msum + wl(n)*ml(n)
           qsum = qsum + wl(n)*ql(n)
           wsum = wsum + wl(n)
           enddo
      if (wsum.lt.2) then
           xbar = 0.
           ybar = 0.
           mbar = 0.
           xsig = 1.
           ysig = 1.
           msig = 1.
           qsig = 1.
           return
           endif
      xbar = xsum/wsum
      ybar = ysum/wsum
      mbar = msum/wsum
      qbar = qsum/wsum
 
      xsum = 0.
      ysum = 0.
      msum = 0.
      qsum = 0.
      asum = 0.
      bsum = 0.
      dmax = 0.
      nmax = 0
      do n = 1, NTOT
           xsum = xsum + wl(n)*(xl(n)-xbar)**2
           ysum = ysum + wl(n)*(yl(n)-ybar)**2
           msum = msum + wl(n)*(ml(n)-mbar)**2
           qsum = msum + wl(n)*(ql(n)-qbar)**2
           asum = asum + wl(n)*((xl(n)-xbar+yl(n)-ybar)/1.41421)**2
           bsum = bsum + wl(n)*((xl(n)-xbar-yl(n)+ybar)/1.41421)**2
           dx = xl(n)-xbar
           dy = yl(n)-ybar
           dd = sqrt(dx**2+dy**2)
           if (dd.gt.dmax.and.wl(n).eq.1) then
              dmax = dd
              nmax = n
              endif
           enddo
      xsig = sqrt(xsum/(wsum-1))
      ysig = sqrt(ysum/(wsum-1))
      msig = sqrt(msum/(wsum-1))
      qsig = sqrt(qsum/(wsum-1))
      dsig = sqrt(xsig**2+ysig**2)
 
      xsum = 0.
      ysum = 0.
      wsumx= 0.
      do n = 1, NTOT
           if (n.ne.nmax) then
              xsum = xsum + wl(n)*xl(n)
              ysum = ysum + wl(n)*yl(n)
              wsumx= wsumx+ wl(n)
              endif
           enddo
      xbarx = xsum/wsumx
      ybarx = ysum/wsumx
      dmaxx = sqrt((xl(nmax)-xbarx)**2+(yl(nmax)-ybarx)**2)
      xsum = 0.
      ysum = 0.
      do n = 1, NTOT
           if (n.ne.nmax) then
              xsum = xsum + wl(n)*(xl(n)-xbarx)**2
              ysum = ysum + wl(n)*(yl(n)-ybarx)**2
              endif
           enddo
      xsigx = sqrt(xsum/(wsumx-1))
      ysigx = sqrt(ysum/(wsumx-1))
      dsigx = sqrt(xsigx**2+ysigx**2)
 
 
c 7/20/2007: the 4*2dsig was waaaay too generous
c
c     if (dmaxx.gt.4.00*dsigx.and.
      if (dmaxx.gt.3.00*dsigx.and.
     .    wsum.gt.NMIN.and.
     .    nmax.gt.0) then
c        print*,'REJECT! '
           wl(nmax) = 0
           goto 1
           endif
 
      NUSED = wsum
 
 
c     if (.not.(xbar.gt.  0.0)) xbar = 0.000
c     if (.not.(ybar.gt.  0.0)) ybar = 0.000
      if (.not.(mbar.gt.-99.0)) mbar = 0.000
      if (.not.(qbar.gt. -0.0)) qbar = 9.999
      if (.not.(xsig.lt.  9.0)) xsig = 9.999
      if (.not.(ysig.lt.  9.0)) ysig = 9.999
      if (.not.(msig.lt.  9.0)) msig = 9.999
      if (.not.(qsig.lt.  9.0)) qsig = 9.999
 
      if (.true.) return
 
      open(19,file='REPORT.BARSIG2.OUT',status='unknown')
      write(19,119) xbar, ybar, -2
      write(19,119) xsig, ysig, -1
      do n = 1, NTOT
           write(19,119) xl(n), yl(n), wl(n), n
  119    format(1x,f10.3,1x,f10.3,1x,i2,1x,i8)
           enddo
      close(19)
 
      return
      end
 
 
c-------------------------------------------------------------
c
c
      subroutine glob_fit6_reju(x1,y1,x2,y2,uu,Ns,Nu,
     .                          AG,BG,CG,DG,Xo,Yo,Wo,Zo,
     .                          dsig,dsig1,dsig2)
      implicit none
      integer Ns, Nu
      real*8  x1(Ns), y1(Ns)
      real*8  x2(Ns), y2(Ns)
      integer uu(Ns)
      real*8 AG,BG,CG,DG
      real*8 Xo,Yo,Wo,Zo
      real    dsig
      real    dsig1, dsig2
 
      real    dx, dy, dd
      integer Nu0, n
 
      real*8 x2e, y2e
 
      real dsigu, dsigf
      integer h, hist(21)
 
      dsigu = dsig1
      dsigf = dsig2
 
      if (Ns.gt._OMAX_) stop 'glob_fit6_rej Ns.gt._OMAX_'
 
      Nu = 0
      do n = 1, Ns
           uu(n) = 1
           enddo
 
  1   continue
      dsigu = dsigu*0.95
 
      call glob_fit6nrDPU(x1,y1,x2,y2,uu,Ns,
     .                    AG,BG,CG,DG,Xo,Yo,Wo,Zo)
 
      do h = 1, 21
           hist(h) = 0
           enddo
 
      dsig  = 0.
      Nu = 0
      do n = 1, Ns
           x2e = AG*(x1(n)-Xo) + BG*(y1(n)-Yo) + Wo
           y2e = CG*(x1(n)-Xo) + DG*(y1(n)-Yo) + Zo
           dx = x2(n)-x2e
           dy = y2(n)-y2e
           dd = sqrt(dx**2+dy**2)
           uu(n) = 1
           if (dd.gt.dsigu) uu(n) = 0
           if (uu(n).eq.1) then
              Nu = Nu + 1
              dsig = dsig + dd
              h  = 1 + dd*10
              if (h.gt.21) h = 21
              hist(h) = hist(h) + 1
              endif
           enddo
      dsig = dsig/Nu
 
c     write(*,119) dsigu,dsigf,dsig,AG,BG,CG,DG,Xo-Wo,Yo-Zo,
c    .             Nu,(hist(h),h=1,21)
  119 format(1x,3f10.4,5x,4f10.5,5x,2f10.3,5x,i6,5x,21i5)
      if (dsigu.gt.dsigf) goto 1
      return
 
      end
 
 
 
      subroutine find_prox_uvm(u_n,v_n,m_n,Ns,d_n)
      implicit none
 
      integer Ns
      real u_n(Ns)
      real v_n(Ns)
      real m_n(Ns)
      real d_n(Ns)
 
      integer Nu
 
      integer, dimension(:,:), allocatable :: hash
      integer NX, NY, RR
 
      real dx, dy, dd
      real umin, umax
      real vmin, vmax
      integer i,ii
      integer j,jj
      integer N,NN
      real ddmin
 
      umin =  99999
      umax = -99999
      vmin =  99999
      vmax = -99999
      do N = 1, Ns
           if (u_n(N).lt.umin) umin = u_n(N)
           if (u_n(N).gt.umax) umax = u_n(N)
           if (v_n(N).lt.vmin) vmin = v_n(N)
           if (v_n(N).gt.vmax) vmax = v_n(N)
           enddo
 
      NX = (umax-umin) + 2
      NY = (vmax-vmin) + 2
 
      allocate(hash(NX,NY))
      do i = 01, NX
      do j = 01, NY
           hash(i,j) = 0
           enddo
           enddo
 
      Nu = 0
      do N = 1, Ns
           i = 1 + (u_n(N)-umin)
           j = 1 + (v_n(N)-vmin)
           if (i.ge.1.and.i.le.NX.and.
     .       j.ge.1.and.j.le.NY) then
               if (hash(i,j).ne.0) then
          NN = hash(i,j)
          if (m_n(NN).gt.m_n(N)) hash(i,j) = 0
          endif
               if (hash(i,j).eq.0) hash(i,j) = N
               endif
           enddo
 
      do N = 1, Ns
           ddmin = 99.00
           i = 1 + (u_n(N)-umin)
           j = 1 + (v_n(N)-vmin)
           do ii = max(i-99,1), min(i+99,NX)
           do jj = max(j-99,1), min(j+99,NY)
              if ((ii-i)**2+(jj-j)**2.le.99.5**2) then
                 if (hash(ii,jj).ne.0) then
            NN = hash(ii,jj)
            if (m_n(NN).lt.m_n(N)) then
               dx = u_n(N)-u_n(NN)
               dy = v_n(N)-v_n(NN)
               dd = sqrt(dx**2+dy**2)
               if (dd.lt.ddmin) ddmin = dd
               endif
            endif
                 endif
              enddo
              enddo
           d_n(N) = ddmin
    3    continue
           enddo
      deallocate(hash)
 
      return
      end
 
 
 
 
 
 
c-----------------------------------------------
c
c
      subroutine xymxym_purge(x1_l,y1_l,m1_l,
     .                        x2_l,y2_l,m2_l,
     .                        uu_l,Lu,Ls,
     .                        AG, BG, CG, DG,
     .                        X1, Y1, X2, Y2,
     .                        mbar,msig,
     .                        dbar , dmax,
     .                        dclip,sclip)  ! input distance clip, sigma clip
      implicit none
 
      integer Ls
      real*8  x1_l(Ls), y1_l(Ls), m1_l(Ls)
      real*8  x2_l(Ls), y2_l(Ls), m2_l(Ls)
      integer uu_l(Ls)
      integer Lu
 
      real*8 AG, BG, CG, DG
      real*8 X1, Y1, X2, Y2
      real   mbar, msig
      real   dbar, dmax
      real   dclip,sclip
 
      real dm_l(9999)
      integer L, Lm, Lmu
      integer Lmax
      integer NIT
      real    dx, dy, dd, dtot
      real*8  x2e, y2e
 
      if (Ls.gt.9999) stop 'Ls.gt.9999'
 
      Lu = Ls
 
      NIT = 0
    1 NIT = NIT + 1
      call glob_fit6nrDPU(x1_l,y1_l,x2_l,y2_l,uu_l,Ls,
     .                    AG,BG,CG,DG,X1,Y1,X2,Y2)
      Lm = 0
      do L = 1, Ls
         if (uu_l(L).eq.1) then
            Lm = Lm + 1
            dm_l(Lm) = m1_l(L)-m2_l(L)
            endif
         enddo
      call rbarsigs(dm_l,Lm,mbar,msig,Lmu,5.0)
      dtot = 0.
      Lmax = 0
      dmax = 0
      Lu   = 0
      do L = 1, Ls
         if (uu_l(L).eq.1) then
            x2e = X2 + AG*(x1_l(L)-X1) + BG*(y1_l(L)-Y1)
            y2e = Y2 + CG*(x1_l(L)-X1) + DG*(y1_l(L)-Y1)
            dx  = x2_l(L)-x2e
            dy  = y2_l(L)-y2e
            dd  = sqrt(dx**2+dy**2)
            dtot = dtot + dd
            Lu   = Lu   + 1
            if (dd.gt.dmax) then
               dmax = dd
               Lmax = L
               endif
            endif
         enddo
      dbar = dtot/Lu
      if (dmax.gt.sclip*dbar.or.dmax.gt.dclip) then
          uu_l(Lmax) = 0
          goto 1
          endif
 
      return
      end
 
 
 
 
      subroutine dswap(x,y)
      implicit none
 
      real*8 x
      real*8 y
      real*8 t
      t = y
      y = x
      x = t
      return
      end
 
 
 
 
 
      subroutine find_hstinfo(NAME,FILT,FILTNUM)
      implicit none
 
      character*9 NAME
      character*5 FILT
      integer     FILTNUM
 
      integer NOBS, NOBSs
      character*5  FILT_OBS(490357)
      character*18 FILTU
      character*9  NAME_OBS(490357)
      character*7  DATE_OBS(490357)
      integer      EXPT_OBS(490357)
      integer       PID_OBS(490357)
      integer      CYNO_OBS(490357)
 
      data NOBSs/0/
 
      character*80 STRING
      character    CHAR
      integer i, ic
 
 
      if (NOBSs.eq.0) then
         open(19,file='readheader_NGC0104_ALL.list',status='old')
         NOBSs = 1782
         do NOBS = 1, 1782
            read(19,'(a80)') STRING
            FILTU = STRING(1:18)
            NAME_OBS(NOBS) = STRING(2:10)
            read(STRING(12:16),*)  PID_OBS(NOBS)
            read(STRING(70:73),*) EXPT_OBS(NOBS)
            FILT_OBS(NOBS) = STRING(63:67)
            enddo
         close(19)
         endif
 
      do NOBS = 1, NOBSs
         if (NAME.eq.NAME_OBS(NOBS)) then
            FILT    = FILT_OBS(NOBS)
            FILTNUM = 0
            if (NAME(1:1).eq.'u'.and.FILT.eq.'F439W') FILTNUM = 01
            if (NAME(1:1).eq.'u'.and.FILT.eq.'F475W') FILTNUM = 02
            if (NAME(1:1).eq.'u'.and.FILT.eq.'F555W') FILTNUM = 03
            if (NAME(1:1).eq.'u'.and.FILT.eq.'F606W') FILTNUM = 04
            if (NAME(1:1).eq.'u'.and.FILT.eq.'F814W') FILTNUM = 05
            if (NAME(1:1).eq.'j'.and.FILT.eq.'F435W') FILTNUM = 06
            if (NAME(1:1).eq.'j'.and.FILT.eq.'F475W') FILTNUM = 07
            if (NAME(1:1).eq.'j'.and.FILT.eq.'F555W') FILTNUM = 08
            if (NAME(1:1).eq.'j'.and.FILT.eq.'F606W') FILTNUM = 09
            if (NAME(1:1).eq.'j'.and.FILT.eq.'F814W') FILTNUM = 10
            if (NAME(1:1).eq.'i'.and.FILT.eq.'F435W') FILTNUM = 11
            if (NAME(1:1).eq.'i'.and.FILT.eq.'F475W') FILTNUM = 12
            if (NAME(1:1).eq.'i'.and.FILT.eq.'F555W') FILTNUM = 13
            if (NAME(1:1).eq.'i'.and.FILT.eq.'F606W') FILTNUM = 14
            if (NAME(1:1).eq.'i'.and.FILT.eq.'F814W') FILTNUM = 15
            endif
         enddo
 
      return
      end
 
 
 
 
      subroutine glob_fit6nrDPU(x1,y1,x2,y2,uu,Ns,
     .                          A,B,C,D,x1o,y1o,x2o,y2o)
      implicit none
 
      real*8 x1(*), y1(*)
      real*8 x2(*), y2(*)
      integer uu(*)
      integer Ns
      real*8 A, B, C, D
      real*8 x1o, y1o, x2o, y2o
 
      real*8 sxx, sx, swx, szx
      real*8 syy, sy, swy, szy
      real*8 sw,  sz, sxy
      real*8 dlta
      real*8 dsxx, dsyy, dsxy
      real*8 dswx, dswy, dszx, dszy
 
      integer n, nnn
 
      if (Ns.lt.3) goto 999
 
      x1o = 0
      y1o = 0
      x2o = 0
      y2o = 0
      nnn = 0
      do n = 1, Ns
           x1o = x1o + uu(n)*x1(n)
           y1o = y1o + uu(n)*y1(n)
           x2o = x2o + uu(n)*x2(n)
           y2o = y2o + uu(n)*y2(n)
           nnn = nnn + uu(n)
           enddo
 
      x1o = x1o/NNN
      y1o = y1o/NNN
      x2o = x2o/NNN
      y2o = y2o/NNN
 
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
           sxy = sxy + uu(n)*(x1(n)-x1o)*(y1(n)-y1o)
           sxx = sxx + uu(n)*(x1(n)-x1o)*(x1(n)-x1o)
           sx  = sx  + uu(n)*(x1(n)-x1o)
           syy = syy + uu(n)*(y1(n)-y1o)*(y1(n)-y1o)
           sy  = sy  + uu(n)*(y1(n)-y1o)
           swx = swx + uu(n)*(x2(n)-x2o)*(x1(n)-x1o)
           swy = swy + uu(n)*(x2(n)-x2o)*(y1(n)-y1o)
           sw  = sw  + uu(n)*(x2(n)-x2o)
           szx = szx + uu(n)*(y2(n)-y2o)*(x1(n)-x1o)
           szy = szy + uu(n)*(y2(n)-y2o)*(y1(n)-y1o)
           sz  = sz  + uu(n)*(y2(n)-y2o)
           enddo
 
 
      dsxx = sx*sx - NNN*sxx
      dsyy = sy*sy - NNN*syy
      dsxy = sx*sy - NNN*sxy
 
      dlta = dsxx*dsyy - dsxy*dsxy
 
      if (dlta.eq.0) goto 999
 
      dswx = sw*sx - NNN*swx
      dswy = sw*sy - NNN*swy
      dszx = sz*sx - NNN*szx
      dszy = sz*sy - NNN*szy
 
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
 
 
 
c-------------------------------------------------------------------------------------
c
cpsf:/user/jayander/MSOTO/02.2MASS_BASIS > head READHEADER/readheader_NGC0104_ALL.list
c.........1.........2.........3.........4.........5.........6.........7.........8.........9
c.........01234567890.........3.........4.........5............34567............8.........9
c j6lk01v5q  9444 09/10/02 08:02:55 00d22m40s0 -72:03:51.8 199 F814W  1390  2.77352 0.0000000  -6.55   0.99 NGC0104
c j6lk01vcq  9444 09/10/02 09:36:32 00d22m40s0 -72:03:51.8 199 F814W  1390  2.77370 0.0000000  -6.55   0.99 NGC0104
c j6lk01vgq  9444 09/10/02 11:12:36 00d22m40s0 -72:03:51.8 199 F814W  1460  2.77388 0.0000000  -6.55   0.99 NGC0104
c j6lk01vlq  9444 09/10/02 12:48:40 00d22m40s0 -72:03:51.8 199 F814W  1460  2.77406 0.0000000  -6.55   0.99 NGC0104
c j6ll01xzq  9443 07/07/02 20:28:51 00d24m04s9 -72:04:50.0  99 F475W    60  2.51740 0.0000000  -0.02   0.02 NGC0104
c
c-------------------------------------------------------------------------------------
 
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/WCS/find_ifilt_ACSWFC.f"
c****                                         
c*********************************************
                                              
 
 
 
      integer function find_ifilt_ACSWFC(FILENAME)
      implicit none
 
      character*(*) FILENAME
 
      character*20 STREAM_FILTNAM1
      character*20 STREAM_FILTNAM2
 
      character*5 FILTNAME(12)
      data FILTNAME/'F435W','F475W','F555W','F606W','F814W',
     .              'F850L','F625W','F658N','F775W','F660N',
     .              'F502N','F550M'/
 
      integer i
 
      find_ifilt_ACSWFC = 0
 
      call query_hdre(FILENAME,'FILTNAM1',STREAM_FILTNAM1,-1)
      call query_hdre(FILENAME,'FILTNAM2',STREAM_FILTNAM2,-1)
 
      find_ifilt_ACSWFC = 4
      do i = 01, 12
         if (FILTNAME(i)(1:5).eq.
     .       STREAM_FILTNAM1(2:6)) find_ifilt_ACSWFC = i
         if (FILTNAME(i)(1:5).eq.
     .       STREAM_FILTNAM2(2:6)) find_ifilt_ACSWFC = i
         enddo
 
      return
      end
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/WCS/find_ifilt_WFC3UV.f"
c****                                         
c*********************************************
                                              
 
 
 
      integer function find_ifilt_WFC3UV(FILENAME)
      implicit none
 
      character*(*) FILENAME
 
      character*20 STREAM_FILTNAME
 
      character*5 FILTNAME(10)
      data FILTNAME/'F275W','F336W','F390W','F438W','F555W',
     .              'F606L','F775W','F814W','F850L','F225W'/
 
      integer i
 
      find_ifilt_WFC3UV = 0
 
      call query_hdre(FILENAME,'FILTNAME',STREAM_FILTNAME,-1)
 
      find_ifilt_WFC3UV = 6
      do i = 01, 10
         if (FILTNAME(i)(1:5).eq.
     .       STREAM_FILTNAME(2:6)) find_ifilt_WFC3UV = i
         enddo
      if (find_ifilt_WFC3UV.eq.10) find_ifilt_WFC3UV = 0
 
      return
      end
 
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/GC/STDGDCs/extract_stdgc.f"
c****                                         
c*********************************************
                                              
 
c------------------------------------------------------
c
c
c
      subroutine extract_stdgc(xr,yr,xc,yc,mc,FILE_STDGC,SENSE)
      implicit none
 
      real*8 xr, yr
      real*8 xc, yc, mc
      character*200 FILE_STDGC
      integer SENSE
 
      save
 
      logical FIRST
      data    FIRST/.true./
 
      character*200 FILE_STDGCu
      character*200 FILE_STDGCc
 
      integer   NDIM_XGC, NDIM_YGC
      integer*4 XGC_0, NDIM_XCG
      integer*4 YGC_0, NDIM_YCG
 
      integer i, ios
      integer k
      character*08   FIELD
      character*20   STREAM
      character*2880 buffc
 
      real    fx,   fy
      real*8  xcu,  ycu
      integer ipix, jpix
 
      integer*4, dimension(:,:), allocatable :: xgc
      integer*4, dimension(:,:), allocatable :: ygc
      integer*2, dimension(:,:), allocatable :: mgc
      integer*4, dimension(:,:), allocatable :: xcg
      integer*4, dimension(:,:), allocatable :: ycg
 
      if (FILE_STDGC(1:4).eq.'NONE') then
         if (SENSE.eq.0) then
            mc = 0.
            return
            endif
         if (SENSE.eq.1) then ! forward...
            xc = xr
            yc = yr
            return
            endif
         if (SENSE.eq.-1) then ! forward...
            xr = xc
            yr = yc
            return
            endif
         endif
 
      if (FIRST) FILE_STDGCc = ' '
 
      FILE_STDGCu = FILE_STDGC
      do k = 200,2,-1
         if (FILE_STDGCu(k:k+4).eq.'.fits')
     .       FILE_STDGCu = FILE_STDGCu(1:k+4)
         enddo
 
      if (FILE_STDGCc.ne.FILE_STDGCu) then
          FILE_STDGCc = FILE_STDGCu
c         print*,'EXTRACT_STDGC --- OPEN(10): ',FILE_STDGCc
          open(10,file=FILE_STDGCc,
     .         status='old',
     .         err   =900,
     .         recl  =2880,
     .         form  ='UNFORMATTED',
     .         access='DIRECT')
          i = 1
          read(10,rec=i,iostat=ios) buffc
          do k = 0, 35
             FIELD  = buffc(k*80+01:k*80+08)
             STREAM = buffc(k*80+11:k*80+30)
             if (FIELD.eq.'NDIM_XGC') read(STREAM,*) NDIM_XGC
             if (FIELD.eq.'NDIM_YGC') read(STREAM,*) NDIM_YGC
             if (FIELD.eq.'NDIM_XCG') read(STREAM,*) NDIM_XCG
             if (FIELD.eq.'NDIM_YCG') read(STREAM,*) NDIM_YCG
             if (FIELD.eq.'XGC_0   ') read(STREAM,*) XGC_0
             if (FIELD.eq.'YGC_0   ') read(STREAM,*) YGC_0
             enddo
          if (.not.FIRST) deallocate(xgc,ygc,mgc,xcg,ycg)
          allocate(xgc(NDIM_XGC,NDIM_YGC))
          allocate(ygc(NDIM_XGC,NDIM_YGC))
          allocate(mgc(NDIM_XGC,NDIM_YGC))
          allocate(xcg(NDIM_XCG,NDIM_YCG))
          allocate(ycg(NDIM_XCG,NDIM_YCG))
          i = 2
          call getext_i4(xgc,NDIM_XGC,NDIM_YGC,i)
          call getext_i4(ygc,NDIM_XGC,NDIM_YGC,i)
          call getext_i2(mgc,NDIM_XGC,NDIM_YGC,i)
          call getext_i4(xcg,NDIM_XCG,NDIM_YCG,i)
          call getext_i4(ycg,NDIM_XCG,NDIM_YCG,i)
          close(10)
          FIRST = .false.
          endif
 
      if (SENSE.eq.1) then ! forward...
          ipix = int(xr)
          jpix = int(yr)
          xc = -9999
          yc = -9999
 
          if (ipix.lt.       1-1) return
          if (ipix.gt.NDIM_XGC+1) return
          if (jpix.lt.       1-1) return
          if (jpix.gt.NDIM_YGC+1) return
 
          if (ipix.lt.1) ipix = 1
          if (jpix.lt.1) jpix = 1
          if (ipix.gt.NDIM_XGC-1) ipix = NDIM_XGC-1
          if (jpix.gt.NDIM_YGC-1) jpix = NDIM_YGC-1
 
          fx = xr-ipix
          fy = yr-jpix
          xc = (1-fx)*(1-fy)*xgc(ipix  ,jpix  )/1d5
     .       + (1-fx)*( fy )*xgc(ipix  ,jpix+1)/1d5
     .       + ( fx )*(1-fy)*xgc(ipix+1,jpix  )/1d5
     .       + ( fx )*( fy )*xgc(ipix+1,jpix+1)/1d5
          yc = (1-fx)*(1-fy)*ygc(ipix  ,jpix  )/1d5
     .       + (1-fx)*( fy )*ygc(ipix  ,jpix+1)/1d5
     .       + ( fx )*(1-fy)*ygc(ipix+1,jpix  )/1d5
     .       + ( fx )*( fy )*ygc(ipix+1,jpix+1)/1d5
          return
          endif
 
      if (SENSE.eq.0) then ! find the zeropoint...
          ipix = int(xr)
          jpix = int(yr)
          mc = 0.000
          if (ipix.lt.         1) return
          if (ipix.gt.NDIM_XGC-1) return
          if (jpix.lt.         1) return
          if (jpix.gt.NDIM_YGC-1) return
          fx = xr-ipix
          fy = yr-jpix
          mc = (1-fx)*(1-fy)*mgc(ipix  ,jpix  )/1d4
     .       + (1-fx)*( fy )*mgc(ipix  ,jpix+1)/1d4
     .       + ( fx )*(1-fy)*mgc(ipix+1,jpix  )/1d4
     .       + ( fx )*( fy )*mgc(ipix+1,jpix+1)/1d4
          return
          endif
 
      if (SENSE.eq.-1) then ! reverse
          xcu =  1 + (xc-XGC_0)
          ycu =  1 + (yc-YGC_0)
          ipix = int(xcu)
          jpix = int(ycu)
          xr = -1
          yr = -1
          if (ipix.lt.         1) return
          if (ipix.gt.NDIM_XCG-1) return
          if (jpix.lt.         1) return
          if (jpix.gt.NDIM_YCG-1) return
          if (xcg(ipix  ,jpix  ).le.-2000000000) return
          if (xcg(ipix+1,jpix  ).le.-2000000000) return
          if (xcg(ipix  ,jpix+1).le.-2000000000) return
          if (xcg(ipix+1,jpix+1).le.-2000000000) return
           fx = xcu-ipix
           fy = ycu-jpix
           xr = (1-fx)*(1-fy)*xcg(ipix  ,jpix  )/1d5
     .        + (1-fx)*( fy )*xcg(ipix  ,jpix+1)/1d5
     .        + ( fx )*(1-fy)*xcg(ipix+1,jpix  )/1d5
     .        + ( fx )*( fy )*xcg(ipix+1,jpix+1)/1d5
           yr = (1-fx)*(1-fy)*ycg(ipix  ,jpix  )/1d5
     .        + (1-fx)*( fy )*ycg(ipix  ,jpix+1)/1d5
     .        + ( fx )*(1-fy)*ycg(ipix+1,jpix  )/1d5
     .        + ( fx )*( fy )*ycg(ipix+1,jpix+1)/1d5
           return
           endif
 
      print*,'ILLEGAL SENSE',SENSE
      stop
 
  900 continue
      print*,'file open error'
      print*,'  FILE_STDGCc: ',FILE_STDGCc
      print*,'  FILE_SGDGC : ',FILE_STDGC
      stop
      end
 
 
c--------------------------------------------------------------------
c
c
      subroutine getext_i4(pix,NDIMX,NDIMY,i)
      implicit none
 
      integer NDIMX, NDIMY
      integer*4 PIX(NDIMX,NDIMY)
      integer i
 
      integer o
      integer nbyte0
      integer nbyteE
      integer nbyte1
      integer nbyte2
      integer nbper
      integer ifirst, i1, i2, ios
      integer np1, np2, npt
      integer nbu, npu
 
      character*2880 buffc
      byte buffb(2880)
      equivalence (buffc,buffb)
 
      byte      b(4)
      integer*4 ii
      equivalence(ii,b)
 
      integer ipix, jpix
 
      ifirst = i+1
      i1 = i
      i2 = i
 
      nbper  = 4*NDIMX*NDIMY
      npt    =   NDIMX*NDIMY
      nbyte1 = 1
      nbyte2 = nbper
      i1 = i+1 + nbyte1/2880
      i2 = i+1 + nbyte2/2880
 
      do i = i1, i2, 1
         nbyte0 = (i-ifirst)*2880+   1
         nbyteE = (i-ifirst)*2880+2880
         np1 = (nbyte0-nbyte1)/4 + 1
         np2 = (nbyteE-nbyte1)/4 + 1
         read(10,rec=i,iostat=ios) buffc
         do o = 1, 720
            npu = np1+o-1
            nbu = (o-1)*4
            ii  = 0
            ipix = npu - (npu-1)/NDIMX*NDIMX
            jpix =   1 + (npu-1)/NDIMX
            if (ipix.ge.1.and.ipix.le.NDIMX.and.
     .          jpix.ge.1.and.jpix.le.NDIMY) then
                if (.not.(_LINUX_)) then
                   b(1) = buffb(nbu+1)
                   b(2) = buffb(nbu+2)
                   b(3) = buffb(nbu+3)
                   b(4) = buffb(nbu+4)
                   endif
                if ((_LINUX_)) then
                   b(4) = buffb(nbu+1)
                   b(3) = buffb(nbu+2)
                   b(2) = buffb(nbu+3)
                   b(1) = buffb(nbu+4)
                   endif
                pix(ipix,jpix) = ii
                endif
            enddo!o
         enddo!i
 
      i = i2 + 1
 
      return
      end
 
 
 
c--------------------------------------------------------------------
c
c
      subroutine getext_i2(pix,NDIMX,NDIMY,i)
      implicit none
 
      integer NDIMX, NDIMY
      integer*2 PIX(NDIMX,NDIMY)
      integer i
 
      integer o
      integer nbyte0
      integer nbyteE
      integer nbyte1
      integer nbyte2
      integer nbper
      integer ifirst, i1, i2, ios
      integer np1, np2, npt
      integer nbu, npu
 
      character*2880 buffc
      byte buffb(2880)
      equivalence (buffc,buffb)
 
      byte      b(2)
      integer*2 ii
      equivalence(ii,b)
 
      integer ipix, jpix
 
      ifirst = i+1
      i1 = i
      i2 = i
 
      nbper  = 2*NDIMX*NDIMY
      npt    =   NDIMX*NDIMY
      nbyte1 = 1
      nbyte2 = nbper
      i1 = i+1 + nbyte1/2880
      i2 = i+1 + nbyte2/2880
 
      do i = i1, i2, 1
         nbyte0 = (i-ifirst)*2880+   1
         nbyteE = (i-ifirst)*2880+2880
         np1 = (nbyte0-nbyte1)/2 + 1
         np2 = (nbyteE-nbyte1)/2 + 1
         read(10,rec=i,iostat=ios) buffc
         do o = 1, 1440
            npu = np1+o-1
            nbu = (o-1)*2
            ii  = 0
            ipix = npu - (npu-1)/NDIMX*NDIMX
            jpix =   1 + (npu-1)/NDIMX
            if (ipix.ge.1.and.ipix.le.NDIMX.and.
     .          jpix.ge.1.and.jpix.le.NDIMY) then
                if (.not.(_LINUX_)) then
                   b(1) = buffb(nbu+1)
                   b(2) = buffb(nbu+2)
                   endif
                if ((_LINUX_)) then
                   b(2) = buffb(nbu+1)
                   b(1) = buffb(nbu+2)
                   endif
                pix(ipix,jpix) = ii
                endif
            enddo!o
         enddo!i
 
      i = i2 + 1
 
      return
      end
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/GC/STDGDCs/xryr2mc_stdgc.f"
c****                                         
c*********************************************
                                              
 
c------------------------------------------------------
c
c mc should be the pixel area correction
c
c
      subroutine xryr2mc_stdgc(xr,yr,mc,FILE_STDGC)
      implicit none
 
      real*8 xr, yr
      real*8 mc
      character*(*) FILE_STDGC
      real*8 xc, yc
 
      call extract_stdgc(xr,yr,xc,yc,mc,FILE_STDGC,0)  ! 0 = pix area correction
 
      return
      end
 
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/GC/STDGDCs/xryr2xcyc_stdgc.f"
c****                                         
c*********************************************
                                              
 
c------------------------------------------------------
c
      subroutine xryr2xcyc_stdgc(xr,yr,xc,yc,FILE_STDGC)
      implicit none
 
      real*8 xr, yr
      real*8 xc, yc
      character*(*) FILE_STDGC
      real*8 mc
 
      call extract_stdgc(xr,yr,xc,yc,mc,FILE_STDGC,1)  ! 1 = forward
 
      return
      end
 
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/GC/STDGDCs/xcyc2xryr_stdgc.f"
c****                                         
c*********************************************
                                              
 
c------------------------------------------------------
c
      subroutine xcyc2xryr_stdgc(xc,yc,xr,yr,FILE_STDGC)
      implicit none
 
      real*8 xc, yc
      real*8 xr, yr
      character*(*) FILE_STDGC
      real*8 mc
 
      call extract_stdgc(xr,yr,xc,yc,mc,FILE_STDGC,-1)  ! -1 = reverse
 
      return
      end
 
 
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/FITSIO/GEN/writfits_i2.f"
c****                                         
c*********************************************
                                              
 
      subroutine writfits_i2(FILE,pix,PXDIMX,PXDIMY)
      implicit none
 
      integer PXDIMX,PXDIMY
      character*(*) FILE
      integer*2 pix(PXDIMX,PXDIMY)
 
      integer nbyte0
      integer nbyteE
      integer nbyte1
      integer nbyte2
      integer nbper
      integer i,ios
 
      character*2880 buffc
      byte buffb(2880)
      equivalence (buffb,buffc)
 
      integer ifirst, i1, i2
 
      integer np1, np2, npt
 
      character*70 HDR(25)
      common/HDR/HDR
 
      character*80 FILEU
 
      FILEU = FILE
      do i = 75,2,-1
         if (FILE(i:i+4).eq.'.fits') FILEU = FILE(1:i+4)
         enddo
 
      !write(*,'(''ENTER   writfits_i2: '',80a)') FILEU
 
      open(10,file=FILEU,
     .     status='unknown',
     .     err   =900,
     .     recl  =2880,
     .     form  ='UNFORMATTED',
     .     access='DIRECT')
 
 
      i = 1
      write(buffc( 0*80+1: 1*80),'(''SIMPLE  = T    '')')
      write(buffc( 1*80+1: 2*80),'(''BITPIX  =  16  '')')
      write(buffc( 2*80+1: 3*80),'(''NAXIS   = '',i12)') 2
      write(buffc( 3*80+1: 4*80),'(''NAXIS1  = '',i12)') PXDIMX
      write(buffc( 4*80+1: 5*80),'(''NAXIS2  = '',i12)') PXDIMY
      write(buffc( 5*80+1: 6*80),'(''DATATYPE= '',9a)')
     .                          ' ''INTEGER*4 '''
      write(buffc( 6*80+1: 7*80),'(''DATE    = '',11a)')
     .                          ' ''28/01/00'''
      write(buffc( 7*80+1: 8*80),'(''BSCALE  = '',i12)') 00001
      write(buffc( 8*80+1: 9*80),'(''BZERO   = '',i12)') 00000
      write(buffc(09*80+1:10*80),'(''CRPIX1  = '',a20)') HDR(01)
      write(buffc(10*80+1:11*80),'(''CRPIX2  = '',a20)') HDR(02)
      write(buffc(11*80+1:12*80),'(''CRVAL1  = '',a20)') HDR(03)
      write(buffc(12*80+1:13*80),'(''CRVAL2  = '',a20)') HDR(04)
      write(buffc(13*80+1:14*80),'(''CTYPE1  = '',a20)') HDR(05)
      write(buffc(14*80+1:15*80),'(''CTYPE2  = '',a20)') HDR(06)
      write(buffc(15*80+1:16*80),'(''CD1_1   = '',a20)') HDR(07)
      write(buffc(16*80+1:17*80),'(''CD1_2   = '',a20)') HDR(08)
      write(buffc(17*80+1:18*80),'(''CD2_1   = '',a20)') HDR(09)
      write(buffc(18*80+1:19*80),'(''CD2_2   = '',a20)') HDR(10)
      write(buffc(19*80+1:20*80),'(''ORIENTAT= '',a20)') HDR(11)
      write(buffc(20*80+1:21*80),'(''PA_APER = '',a20)') HDR(12)
      write(buffc(21*80+1:22*80),'(''PA_V3   = '',a20)') HDR(13)
      write(buffc(22*80+1:23*80),'(''DATE-OBS= '',a20)') HDR(14)
      write(buffc(23*80+1:24*80),'(''TIME-OBS= '',a20)') HDR(15)
      write(buffc(24*80+1:25*80),'(''EXPTIME = '',a20)') HDR(16)
      write(buffc(25*80+1:26*80),'(''ROOTNAME= '',a20)') HDR(17)
      write(buffc(26*80+1:27*80),'(''TARGNAME= '',a20)') HDR(18)
      write(buffc(27*80+1:28*80),'(''RA_TARG = '',a20)') HDR(19)
      write(buffc(28*80+1:29*80),'(''DEC_TARG= '',a20)') HDR(20)
      write(buffc(29*80+1:30*80),'(''PROPOSID= '',a20)') HDR(21)
      write(buffc(30*80+1:31*80),'(''FILTER1 = '',a20)') HDR(22)
      write(buffc(31*80+1:32*80),'(''FILTER2 = '',a20)') HDR(23)
      write(buffc(32*80+1:33*80),'(''VAFACTOR= '',a20)') HDR(24)
      write(buffc(33*80+1:34*80),'(''COMMENT   '')')
      write(buffc(34*80+1:35*80),'(''COMMENT   '')')
      write(buffc(35*80+1:36*80),'(''END       '')')
 
      write(10,rec=i,iostat=ios) buffc
 
      ifirst = i+1
      i1 = i
      i2 = i
 
      nbper  = 2*PXDIMX*PXDIMY
      npt    =   PXDIMX*PXDIMY
      nbyte1 = 1
      nbyte2 = nbper
      i1 = i+1 + nbyte1/2880
      i2 = i+1 + nbyte2/2880
 
 
      do i = i1, i2, 1
         nbyte0 = (i-ifirst)*2880+   1
         nbyteE = (i-ifirst)*2880+2880
         np1 = (nbyte0-nbyte1)/4 + 1
         np2 = (nbyteE-nbyte1)/4 + 1
         np1 = (nbyte0-nbyte1)/2 + 1
         np2 = (nbyteE-nbyte1)/2 + 1
        !if (i.lt.0010) print*,'i: ',i,i1,i2,np1,np2,nbyte0
        !if (i.gt.2900) print*,'i: ',i,i1,i2,np1,np2,nbyte0
         call pix2buff_i2(buffb,pix,np1,npt)
         write(10,rec=i,iostat=ios) buffc
         enddo
 
      close(10)
 
      return
 
 900  continue
      print*,'WRITFITS.f ERROR'
      stop
 
      end
 
 
      subroutine pix2buff_i2(buff,pix,n1,nt)
      implicit none
      byte buff(2880)
      integer*2 pix(*)
      integer n1,nt
 
      byte b(2)
      integer*2 ii
      equivalence(ii,b)
 
      integer i, npu, nbu
      logical islinux
 
      do i = 1, 1440
         npu = n1+i-1
         nbu = (i-1)*2
         if (npu.ge.1.and.npu.le.nt) ii = pix(npu)
         if (.not.(islinux())) then
            buff(nbu+1) = b(1)
            buff(nbu+2) = b(2)
           else
            buff(nbu+1) = b(2)
            buff(nbu+2) = b(1)
            endif
         enddo
 
      return
      end
 
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/FITSIO/GEN/writfits_r4_WCSLTV.f"
c****                                         
c*********************************************
                                              
 
 
c-----------------------------------------------------
c
c this just writes a real*4 fits image
c
 
      subroutine writfits_r4_WCSLTV(FILE,pix,PXDIMX,PXDIMY,
     .                              CRPIX1,CRPIX2,CRVAL1,CRVAL2,
     .                              CD1_1, CD1_2, CD2_1, CD2_2,
     .                              LTV1, LTV2,
     .                              LTM1_1,LTM1_2,LTM2_1,LTM2_2)
      implicit none
 
      character*(*) FILE
      integer PXDIMX,PXDIMY
      real    pix(PXDIMX,PXDIMY)
      real*8  CRPIX1,CRPIX2,CRVAL1,CRVAL2
      real*8  CD1_1, CD1_2, CD2_1, CD2_2
      real*8  LTV1,  LTV2
      real*8  LTM1_1,LTM1_2,LTM2_1,LTM2_2
 
 
      integer nbyte0
      integer nbyteE
      integer nbyte1
      integer nbyte2
      integer nbper
      integer i,ios
 
      character*2880 buffc
      byte buffb(2880)
      equivalence (buffb,buffc)
 
      integer ifirst, i1, i2
 
      integer np1, np2, npt
      integer k
 
      character*80 FILEU
      character*70 HDR(25)
      common/HDR/HDR
 
      FILEU = FILE
      do i = 75,2,-1
         if (FILE(i:i+4).eq.'.fits') FILEU = FILE(1:i+4)
         enddo
 
      open(10,file=FILEU,status='unknown',
     .     err=900,recl=2880,form='UNFORMATTED',
     .     access='DIRECT')
 
      do k = 00, 34
         write(buffc(k*80+1:k*80+80),'(80x)')
         enddo
 
      write(10,rec=i,iostat=ios) buffc
 
      write(buffc(00*80+1:01*80),'(''SIMPLE  =                    T'')')
      write(buffc(01*80+1:02*80),'(''BITPIX  =                  -32'')')
      write(buffc(02*80+1:03*80),'(''NAXIS   = '',8x,i12)') 2
      write(buffc(03*80+1:04*80),'(''NAXIS1  = '',8x,i12)') PXDIMX
      write(buffc(04*80+1:05*80),'(''NAXIS2  = '',8x,i12)') PXDIMY
      write(buffc(05*80+1:06*80),'(''DATATYPE= '',9a)')  " 'REAL*4' "
      write(buffc(06*80+1:07*80),'(''COMMENT   '',a05)') '     '
      write(buffc(07*80+1:08*80),'(''CTYPE1  = '',a10)') "'RA---TAN'"
      write(buffc(08*80+1:09*80),'(''CTYPE2  = '',a10)') "'DEC--TAN'"
      write(buffc(09*80+1:10*80),'(''CRPIX1  = '',f20.10)') CRPIX1
      write(buffc(10*80+1:11*80),'(''CRPIX2  = '',f20.10)') CRPIX2
      write(buffc(11*80+1:12*80),'(''CRVAL1  = '',f20.10)') CRVAL1
      write(buffc(12*80+1:13*80),'(''CRVAL2  = '',f20.10)') CRVAL2
      write(buffc(13*80+1:14*80),'(''CD1_1   = '',f20.10)') CD1_1
      write(buffc(14*80+1:15*80),'(''CD1_2   = '',f20.10)') CD1_2
      write(buffc(15*80+1:16*80),'(''CD2_1   = '',f20.10)') CD2_1
      write(buffc(16*80+1:17*80),'(''CD2_2   = '',f20.10)') CD2_2
      write(buffc(17*80+1:18*80),'(''COMMENT   '',a05)') '     '
      write(buffc(18*80+1:19*80),'(''LTV1    = '',f20.10)') LTV1
      write(buffc(19*80+1:20*80),'(''LTV2    = '',f20.10)') LTV2
      write(buffc(20*80+1:21*80),'(''LTM1_1  = '',f20.10)') LTM1_1
      write(buffc(21*80+1:22*80),'(''LTM1_2  = '',f20.10)') LTM1_2
      write(buffc(22*80+1:23*80),'(''LTM2_1  = '',f20.10)') LTM2_1
      write(buffc(23*80+1:24*80),'(''LTM2_2  = '',f20.10)') LTM2_2
      write(buffc(24*80+1:25*80),'(''COMMENT   '',a05)') '     '
      write(buffc(25*80+1:26*80),'(''END       '')')
 
      write(10,rec=i,iostat=ios) buffc
 
 
      ifirst = i+1
      i1 = i
      i2 = i
 
      nbper  = 4*PXDIMX*PXDIMY
      npt    =   PXDIMX*PXDIMY
      nbyte1 = 1
      nbyte2 = nbper
      i1 = i+1 + nbyte1/2880
      i2 = i+1 + nbyte2/2880
 
      do i = i1, i2, 1
         nbyte0 = (i-ifirst)*2880+   1
         nbyteE = (i-ifirst)*2880+2880
         np1 = (nbyte0-nbyte1)/4 + 1
         np2 = (nbyteE-nbyte1)/4 + 1
         call pix2buff_r4_WCS(buffb,pix,np1,npt)
         write(10,rec=i,iostat=ios) buffc
         enddo
 
      close(10)
 
      return
 
 900  continue
      print*,'writfits_r4.f ERROR'
      print*,'   FILEU: ',FILEU
      stop
 
      end
 
c-------------------------------------------------------
c
c
      subroutine pix2buff_r4_WCS(buff,pix,n1,nt)
      implicit none
      byte buff(2880)
      real*4 pix(*)
      integer n1,nt
 
      byte b(4)
      real*4 r
      equivalence(r,b)
 
      integer i, npu, nbu
 
      do i = 1, 720
         npu = n1+i-1
         nbu = (i-1)*4
         if (npu.ge.1.and.npu.le.nt) r = pix(npu)
         if (.not.(_LINUX_)) then
            buff(nbu+1) = b(1)
            buff(nbu+2) = b(2)
            buff(nbu+3) = b(3)
            buff(nbu+4) = b(4)
            endif
         if ((_LINUX_)) then
            buff(nbu+1) = b(4)
            buff(nbu+2) = b(3)
            buff(nbu+3) = b(2)
            buff(nbu+4) = b(1)
            endif
         enddo
 
      return
      end
 
 
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/HST1PASS/ROUTINES/sub_hst2xym_info.f"
c****                                         
c*********************************************
                                              
 
c*********************************************
c****
c**** #include "/user/jayander/FORTRAN/HST1PASS/ROUTINES/sub_hst2xym_info.f"
c****
c*********************************************
 
 
c
c modify it to ignore saturated CRs
c
 
c
c modify it to deal with drz images...
c
c This is the workhorse routine that finds stars and measures them
c in generic HST images.
c
c
      subroutine sub_hst2xym(HH,HHo,HHs,
     .                       HMIN,FMIN,PMAX,
     .                       HIFLAG,WCSMODE,DOSATD,
     .                       SHOW_ALL,SHOW_USE,SHOW_FND,
     .                       SHOW_REF,SHOW_SUB,SHOW_MSK,
     .                       NPERTs,
     .                       KSEL,                         ! KSEL added 2017.07.18
     .                       QMAX,CMIN,CMAX,QCAND,         ! CMIN, CMAX, QCAND added 2016.06.02
     .                       IMIN,IMAX,JMIN,JMAX,DOMSORT,DOHSORT,
     .                       CATMAT, CATREG,
     .                       FILENAME,NIMu,NIMs,
     .                          PSFFILE_USE,
     .                          GDCFILE_USE,
     .                       Ns,  u_n, v_n,mm_n,
     .                            x_n, y_n, m_n, k_n,
     .                           xx_n,yy_n,
     .                           uu_n,vv_n,
     .                            h_n,hh_n,
     .                            r_n, d_n,
     .                            i_n, j_n, p_n,
     .                            q_n, c_n, s_n,
     .                       BDRY_XR, BDRY_YR,
     .                       BDRY_XC, BDRY_YC,
     .                       BDRY_UG, BDRY_VG,
     .                       BDRY_RA, BDRY_DE,
     .                       BDRY_UU, BDRY_VV,
     .                       FILTu,EXPTu,RDATu,PROPu,INSTu,
     .                       CRPIX1_DEST,CRPIX2_DEST,
     .                       CRVAL1_DEST,CRVAL2_DEST,
     .                       CD1_1_DEST, CD1_2_DEST,
     .                       CD2_1_DEST, CD2_2_DEST,
     .                       VERBOSE,
     .                       As,xinp_a,yinp_a,minp_a,
     .                       TEST1,FOCUS_LEVELi,DOSTARDB)
      implicit none
 
      character*249 HH(999)
      integer       HHo
      integer       HHs
      integer       HMIN
      real          FMIN
      real          PMAX
      character*080 PSFFILE_USE
      character*080 GDCFILE_USE
      integer       HIFLAG
      character*080 WCSMODE
      logical       DOSATD
      character*080 SHOW_ALL
      character*080 SHOW_USE
      character*080 SHOW_FND
      character*080 SHOW_REF
      character*080 SHOW_SUB
      character*080 SHOW_MSK
      integer       NPERTs
      integer       KSEL
      real          QMAX
      real          CMIN, CMAX
      logical       QCAND
      integer       IMIN, IMAX
      integer       JMIN, JMAX
      logical       DOHSORT
      logical       DOMSORT
      character*080 CATMAT
      character*080 CATREG
      character*200 FILENAME
      integer       NIMu
      integer       NIMs
      real*8         u_n(_NSTMAX_),  v_n(_NSTMAX_), mm_n(_NSTMAX_)
      real*8         x_n(_NSTMAX_),  y_n(_NSTMAX_),  m_n(_NSTMAX_)
      real*8        xx_n(_NSTMAX_), yy_n(_NSTMAX_)
      real*8        uu_n(_NSTMAX_), vv_n(_NSTMAX_)
      integer        k_n(_NSTMAX_)
      integer        h_n(_NSTMAX_)
      real          hh_n(_NSTMAX_)
      real*8         r_n(_NSTMAX_),  d_n(_NSTMAX_)
      integer        i_n(_NSTMAX_),  j_n(_NSTMAX_)
      real           p_n(_NSTMAX_)
      real           q_n(_NSTMAX_),  c_n(_NSTMAX_)
      real           s_n(_NSTMAX_)
      integer      sat_n(_NSTMAX_)
      real*8       BDRY_XR(4,4), BDRY_YR(4,4)
      real*8       BDRY_XC(4,4), BDRY_YC(4,4)
      real*8       BDRY_UG(4,4), BDRY_VG(4,4)
      real*8       BDRY_RA(4,4), BDRY_DE(4,4)
      real*8       BDRY_UU(4,4), BDRY_VV(4,4)
      character*05 FILTu
      real         EXPTu
      real         RDATu
      integer      PROPu
      integer      INSTu
      real*8  CRPIX1_DEST, CRPIX2_DEST
      real*8  CRVAL1_DEST, CRVAL2_DEST
      real*8   CD1_1_DEST,  CD1_2_DEST
      real*8   CD2_1_DEST,  CD2_2_DEST
      integer VERBOSE
      integer As
      real*8  xinp_a(As)
      real*8  yinp_a(As)
      real*8  minp_a(As)
      integer TEST1
      real    FOCUS_LEVELi
      logical DOSTARDB
c
c----------------------------------------------------------------------
c
      real          FOCUS_LEVELu
      character*200 STRING
      character*080 PROGNAME
      character*080 TEMPSTRING
      character*080 PSFFILE_INP
      character*080 PSFFILE_PFF
      character*080 PSFFILE_LIB
      character*080 PSFFILE_PRT
      character*080 FILEFITS
      character*080 DIRECT
      character*080 PREFIX
      character*9   STEM, STEMu
      character*7   SUFFIX
      character*080 FILEOUT, FILEu
 
      character*9   STEMs_N(_NIMMAX_)
      common/STEMs_N/STEMs_N
 
      character*080 GDCFILE_INP
      character*080 GDCFILE_LIB
 
      integer*4 NAXIS1
      integer*4 NAXIS2
 
 
      character*80 SHOW_USEu
      character*80 SHOW_FNDu
 
      integer PREFIX_LEN
      integer i0, j0
      integer ii, jj
      integer  i,  j
      integer iu
      integer NSAT, NUNSAT
      logical SATD, trip
      integer ir
 
      real qr
      real cr
      real dx,dy
      integer hobs, fnd_hloc_NAXIS
      integer pobs, fnd_ploc_NAXIS
 
      real crat
      common /crat/ crat
      integer mmeth
      common /mmeth/ mmeth
 
      integer NARGX, NARGY, NARGu
      integer iargc, NARG, NARGs
      integer N, Ns, Nx, NN
 
      integer      ISIMG_NARG(999)
      character*80 STRNG_NARG(999)
 
      integer BITPIX
      common /BITPIX_/BITPIX
 
      real    xr4,yr4, sr4, fr4, ysigfit
      real    xg4,yg4
      integer kr, kru
 
      real*8  xr, yr, fr, sr, mr, dmc
      real*8  dr
      real*8  xro,yro,fro,sro
      real*8  dclip
      real    z_xyoptXX_NAXIS
      real    apl(10)
      real    fadd, fsig, fnoi, noise_sig
      real*8  faddt, faddn
 
      integer ixr0, iyr0
      real*8 xgcu, ygcu
      real*8 xgc,  ygc
      real*8 xgc0, ygc0
      real*8 mgc, m1k, mzp
      real   wfc_zpa
      real   wfc3uv_zpa
 
      real*8 RA, DE
      real*8 ug, vg
      real*8 uu, vv
 
      real xtot, ytot, ptot, ftot
 
      real ss
      real rpsf_phot
      real HMB, histmode_NAXIS
      real mbar_sky_NAXIS
      real mbar_skyopt_NAXIS
      real minsky
 
      character*11 ra2sexig
      character*12 de2sexig
c
c the inner and outer radii used to find
c sky
c
      integer irmin, irmax
      common /SKYINFO_/irmin,irmax
 
      integer hu,  hhist(16)
      character*20 hhwhy(16)
      data hhwhy/
     .     '..1..not on chip....',
     .     '..2..not loc max....',
     .     '..3..nearby brighter',
     .     '..4..flux too low...',
     .     '..5..below mask thr.',
     .     '..6..out of bounds..',
     .     '..7..loflag issues..',
     .     '..8..saturated......',
     .     '..9..not contig sat.',
     .     '.10..not loc sat max',
     .     '.11..sat weirdness..',
     .     '.12..pk but not qsel',
     .     '.13..pk too peaky...',
     .     '.14..satd CR........',
     .     '.15..good unsat.....',
     .     '.16..good satd......'/
 
      real apphot_NAXIS
 
      real*4,    dimension(:,:),   allocatable :: pixo
      real*4,    dimension(:,:),   allocatable :: pixp
      real*4,    dimension(:,:),   allocatable :: pixx
      real*4,    dimension(:,:),   allocatable :: pixc
      real*4,    dimension(:,:),   allocatable :: pixw
      byte  ,    dimension(:,:),   allocatable :: pixq
      integer*4, dimension(:,:),   allocatable :: pixn
 
      real*4, dimension(:,:),   allocatable :: pixm
      real*4, dimension(:,:),   allocatable :: pixs
      real*4, dimension(:,:),   allocatable :: pixr
      byte  , dimension(:,:),   allocatable :: pixy
 
      real*4, dimension(:,:,:), allocatable :: pix_3D
 
      real pixe, pixc0
 
      real fout, fxout, fyout
      real fest, festr
 
      real zzz
      common /zzz/zzz
      real reff
      common /reff/reff
 
      logical DIAG
      common /DIAG_/DIAG
      data    DIAG/.false./
 
      logical DOAPPHOT
      real    RAP
      integer SKI,SKO
 
 
      character*80 IMSUBu
      character*80 IMREFu
      character*80 IMMSKu
 
      integer NSATGLOM
 
c-----------------------------------------
c
c contains info from the fits image header
c
      character*70 HDR(25)
      common/HDR/HDR
      character*70 INFO(10)
      common / fitsinfo / INFO
      integer GAIN
      data GAIN/ 0 /
      real   rGAIN
 
 
c----------------------------------
c
c here are the 9x9 fiducial PSFs
c
 
      real psfarr1(10,10)
      real psfarr2(10,10)
      real psfloc(101,101)
 
 
      real psfpert(101,101,_NPRTMX_,_NPRTMX_,_NIMMXP_)
 
 
      character*8  FIELD
      character*20 STREAM
 
      real*4  EXPT_EFF
 
      integer NFILT ! the mapping for Mario's program
 
      integer iit, jjt
      integer ninput
      data ninput/0/
 
      character*20 DATESTR
      character*20 TIMESTR
 
      real   rDAT
      real*8 rDATuu
 
 
      integer NXPSFs, NYPSFs
      integer ilist(10)
      integer jlist(10)
 
      integer LNC ! LAST NON CHARACTER
      integer H
 
      real*8  CRPIX1_OUTP, CRPIX2_OUTP
      real*8  CRVAL1_OUTP, CRVAL2_OUTP
      real*8   CD1_1_OUTP,  CD1_2_OUTP
      real*8   CD2_1_OUTP,  CD2_2_OUTP
      real*8  RCD1_1_OUTP, RCD1_2_OUTP
      real*8  RCD2_1_OUTP, RCD2_2_OUTP
 
      integer PX, PY
      integer ipsf, jpsf
 
      integer NSATs
      integer isat(_NSTMAX_)
      integer jsat(_NSTMAX_)
      real    zsat(_NSTMAX_)
 
 
c-------------------------------------------------------------------
c
c for find_obj
c
      real*8 rRA , rDE      ! input
      character*11 OBJ_NAME ! output (nearest obj)
      real*8  RA0,  DE0     ! output (locn)
      real*8 dRA0, dDE0
      real*8  dRA,  dDE
      real*8  rd2x, rd2y
      real*8  xy2r, xy2d
 
 
      character*5 FILTNAME
      character*80 FILT1, FILT2, FILT, FILT3, FILT4
      integer PID
 
c
c----------------------------------------------------------------
c
 
 
      integer om_n(_NSTMAX_), od_n(_NSTMAX_)
      real    su_n(_NSTMAX_)
 
      real         dd, dm, rr
      integer      F
      character*20 CHAROUT
      character*20 CHAROUT_F(_FIELDs_)
      character*20 CHAROUT_N(_NSTMAX_)
 
      character*300   STRING300, STRING300H
      integer         F_NITEM(99)
      character*4     CNITEM
      integer         CLAST
      integer         C1_F(_FIELDs_), N1_NITEM(99), N1
      integer         C2_F(_FIELDs_), N2_NITEM(99), N2
      character*23 CHAR_HEADER(_FIELDs_)
      data  CHAR_HEADER / 'u -- u ref-frame       ', ! 01
     .                    'v -- v ref-frame       ', ! 02
     .                    'M -- instl mag (1000s) ', ! 03
     .                    'x -- raw detector x pos', ! 04
     .                    'y -- raw detector y pos', ! 05
     .                    'm -- instl mag         ', ! 06
     .                    'k -- detector chip no  ', ! 07
     .                    'X -- dist-cor x posn   ', ! 08
     .                    'Y -- dist-cor y posn   ', ! 09
     .                    'h -- local isolation   ', ! 10
     .                    'H -- list-based isolatn', ! 11
     .                    'r -- RA (deg)          ', ! 12
     .                    'd -- Dec (deg)         ', ! 13
     .                    'R -- RA (sexig)        ', ! 14
     .                    'D -- Dec (sexig)       ', ! 15
     .                    'i -- peak pixel column ', ! 16
     .                    'j -- peak pixel row    ', ! 17
     .                    'p -- peak pixel value  ', ! 18
     .                    'q -- quality of fit    ', ! 19
     .                    'c -- central excess    ', ! 20
     .                    'U -- u ref-frame (corr)', ! 21
     .                    'V -- v ref-frame (corr)', ! 22
     .                    'k -- chip number       ', ! 23
     .                    'K -- chip number (w/K) ', ! 24
     .                    's -- sky value         ', ! 25
     .                    'n -- star number       '/ ! 26
 
      integer       NITEM
      integer       NLIST
      integer       NREG
 
c
c----------------------------------------------------------------
c
 
      integer K, P, Ks, Ku
      integer UMIN, UMAX
      integer VMIN, VMAX
      integer NAXISU, NAXISV
 
      real*8      RA_MIN, RA_MAX
      real*8      DE_MIN, DE_MAX
 
      integer     O, Os
      real*8      ra_o(_OMAX_)
      real*8      de_o(_OMAX_)
      real*8      mj_o(_OMAX_)
      real*8      mh_o(_OMAX_)
      real*8      mk_o(_OMAX_)
      character*3 aa_o(_OMAX_)
      integer     kk_o(_OMAX_)
      real*8      ug_o(_OMAX_)
      real*8      vg_o(_OMAX_)
 
      integer     U, Us
      integer     oo_u(_OMAX_)
      real*8      ug_u(_OMAX_)
      real*8      vg_u(_OMAX_)
      real*8      mj_u(_OMAX_)
 
      logical     inside_poly
 
      integer MSORT, MSORTs
      real*8  u_msort(_NSTMAX_)
      real*8  v_msort(_NSTMAX_)
      real*8  m_msort(_NSTMAX_)
      integer n_msort(_NSTMAX_)
 
      real*8  AG,BG,CG,DG
      real*8  GA,GB,GC,GD
      real*8  u1o,v1o,u2o,v2o
      integer Lu, Ls
      real    dbar, dmax
      real*8  u1mat(_OMAX_), v1mat(_OMAX_), m1mat(_OMAX_)
      real*8  u2mat(_OMAX_), v2mat(_OMAX_), m2mat(_OMAX_)
      integer n1mat(_OMAX_)
      integer n2mat(_OMAX_)
      integer NMAT, NMATs
      integer n_NMAT(_OMAX_)
      integer o_NMAT(_OMAX_)
 
      integer A
 
      character*20 STRING20_INSTRUME
      character*20 STRING20_DETECTOR
 
      integer NAXIS_PSF
      integer i4_query_hdre
      real*4  r4_query_hdre
 
c------------------------------------------
c INSTRUMENT-BASED INFORMATION
c
 
      real*8   CRPIX1_INST,  CRPIX2_INST
      real*8   CRVAL1_INST,  CRVAL2_INST
      real*8   CD1_1_INST,   CD1_2_INST
      real*8   CD2_1_INST,   CD2_2_INST
      real*8  RCD1_1_INST,  RCD1_2_INST
      real*8  RCD2_1_INST,  RCD2_2_INST
 
      integer LOFLAG_INST
      integer HIFLAG_INST
      character*05 FILT_INST
      real*4  EXPT_INST
      real*4  RDAT_INST
      integer PROP_INST
 
      integer PMAX_PERT
      real    EXPTi
 
      logical ISDRZ
      integer neq
      integer NFOCu
      integer*4 qqqq
 
      integer NSATx
 
      real cexp
      real atot, aexp
      real btot, bexp
      real xru,  yru
 
      real FMINu
 
      real g1, g2
      integer iii, jjj
      real pabs
 
 
      integer HIFLAG_COMMON
      common / HIFLAG_ / HIFLAG_COMMON
 
      integer LOFLAG_COMMON
      common / LOFLAG_ / LOFLAG_COMMON
 
c
c------------------------------------------
 
      FMINu    = FMIN
      DOAPPHOT = .false.
      ISDRZ    = .false.
 
      PSFFILE_PFF = PSFFILE_USE
      PMAX_PERT = 0
 
      write(*,993) FILENAME(1:60)
  993 format('enter subroutine sub_hst2xym: FILENAME = ',60a)
 
      if (PSFFILE_USE(1:6).eq.'APPHOT') then
          if (VERBOSE.gt.0) then
              write(*,'(''#PSFFILEU: '',80a)') PSFFILE_USE
              endif
          read(PSFFILE_USE(7:80),*) RAP,SKI,SKO
          if (VERBOSE.gt.0) then
              write(*,'(''# ---> APERTURE RAP: '',f8.4)') RAP
              write(*,'(''# --->    INNER SKI: '',i3)') SKI
              write(*,'(''# --->    OUTER SKO: '',i3)') SKO
              endif
          DOAPPHOT = .true.
          if (VERBOSE.gt.0) then
              print*,'DOAPPHOT: ',DOAPPHOT
              endif
          endif
 
      if (TEST1.ne.1) then
         print*,'TEST1: ',TEST1
         stop 'sub_hst2xym problem...'
         endif
 
      if (VERBOSE.ge.1) then
          print*,' '
          write(*,'(''ENTER sub_hst2xym -- '',50a)') FILENAME(1:50)
          write(*,'(''ENTER sub_hst2xym -- As: '',i6)') As
          write(*,'(''ENTER sub_hst2xym --  V: '',i6)') VERBOSE
          print*,' '
          endif
 
      PSFFILE_INP = PSFFILE_USE
      GDCFILE_INP = GDCFILE_USE
 
      if (VERBOSE.gt.0) print*,'DOAPPHOT: ',DOAPPHOT
 
      if (VERBOSE.ge.2) then
         print*,'        HHo: ',HHo
         do H = 1, HHo
            write(*,'(15x,i4.4,1x,a80)') H, HH(H)(1:80)
            enddo
         print*,'       HMIN: ',HMIN
         print*,'       FMIN: ',FMIN
         print*,'      FMINu: ',FMINu
         print*,'       PMAX: ',PMAX
         print*,'PSFFILE_INP: ',PSFFILE_INP
         print*,'GDCFILE_INP: ',GDCFILE_INP
         print*,'     HIFLAG: ',HIFLAG
         print*,'    WCSMODE: ',WCSMODE
         print*,'     DOSATD: ',DOSATD
         print*,'   SHOW_ALL: ',SHOW_ALL
         print*,'   SHOW_USE: ',SHOW_USE
         print*,'   SHOW_FND: ',SHOW_FND
         print*,'   SHOW_REF: ',SHOW_REF
         print*,'   SHOW_SUB: ',SHOW_SUB
         print*,'   SHOW_MSK: ',SHOW_MSK
         print*,'     NPERTs: ',NPERTs
         print*,' FOCUS_LEVi: ',FOCUS_LEVELi
         print*,'       KSEL: ',KSEL
         print*,'       QMAX: ',QMAX
         print*,'       CMIN: ',CMIN,CMAX
         print*,'      QCAND: ',QCAND
         print*,'       IMIN: ',IMIN,IMAX
         print*,'       JMIN: ',JMIN,JMAX
         print*,'    DOHSORT: ',DOHSORT
         print*,'    DOMSORT: ',DOMSORT
         print*,'     CATMAT: ',CATMAT
         print*,'     CATREG: ',CATREG
         print*,'   FILENAME: ',FILENAME
         print*,'       NIMu: ',NIMu
         print*,'       NIMs: ',NIMs
         print*,'             '
         endif
 
      if (VERBOSE.ge.2) print*,'---> HHo: ',HHo
      do H = HHo+1, 999  ! clear the rest of the header
         write(HH(H),'(''#'',248('' ''))')
         enddo
      HHs = HHo
 
      OBJ_NAME = ' '
      rRA = 0.
      RA0 = 0.
      dRA = 0.
      dRA = 0.
      rDE = 0.
      RDE = 0.
      dDE = 0.
      dDE = 0.
 
      FILTNAME = 'NONE'
 
      if (VERBOSE.ge.2) print*,'---> dirstrip filename...'
      call dirstrip(FILENAME,DIRECT,PREFIX,SUFFIX)
      STEM  = PREFIX(1:9)
      STEMu = STEM
      STEMs_N(NIMu) = STEMu
 
 
      FILTNAME = 'NONE'
 
      if (VERBOSE.ge.2) print*,'---> dirstrip filename...'
      call dirstrip(FILENAME,DIRECT,PREFIX,SUFFIX)
      STEM  = PREFIX(1:9)
      STEMu = STEM
      STEMs_N(NIMu) = STEMu
 
      if (VERBOSE.ge.2) print*,'---> querY...'
      call  query_hdre_i4(FILENAME,'PROPOSID',PID,-1)
      print*,'    PID: ',PID
      call  query_hdre_r4(FILENAME,'EXPTIME ',EXPTi,-1)
      print*,'  EXPTi: ',EXPTI
      call  query_hdre(FILENAME,'FILTER1 ',FILT1,-1)
      print*,'  EXPTi: ',FILT1
      call  query_hdre(FILENAME,'FILTER2 ',FILT2,-1)
      print*,'  EXPTi: ',FILT2
      call  query_hdre(FILENAME,'FILTNAM1',FILT3,-1)
      call  query_hdre(FILENAME,'FILTNAM2',FILT4,-1)
      call  query_hdre(FILENAME,'FILTER  ',FILT ,-1)
 
      if (VERBOSE.ge.2) print*,'---> find_rdate...'
      call find_rdate(FILENAME,rDAT,TIMESTR,DATESTR)
      rDATuu = rDAT
      if (rDAT.ge.050.and.rDAT.lt.200) rDATuu = rDAT + 1990.0d0
      if (rDAT.le.050)                 rDATuu = rDAT + 2000.0d0
 
      if (VERBOSE.ge.2) print*,'---> rDATuu...',rDATuu
 
      FILTNAME = FILT
      if (FILT1(1:1).eq.'F')  FILTNAME = FILT1(1:5)
      if (FILT2(1:1).eq.'F')  FILTNAME = FILT2(1:5)
      if (FILT3(1:1).eq.'F')  FILTNAME = FILT3(1:5)
      if (FILT4(1:1).eq.'F')  FILTNAME = FILT4(1:5)
      if (FILT1(2:2).eq.'F')  FILTNAME = FILT1(2:6)
      if (FILT2(2:2).eq.'F')  FILTNAME = FILT2(2:6)
      if (FILT3(2:2).eq.'F')  FILTNAME = FILT3(2:6)
      if (FILT4(2:2).eq.'F')  FILTNAME = FILT4(2:6)
 
      if (VERBOSE.ge.2) print*,'---> FILTNAME...'
      EXPTu = EXPTi
      FILTu = FILTNAME(1:5)
      PROPu = PID
      RDATu = rDATuu
 
 
 
      if (VERBOSE.ge.2) print*,'---> _NPRTMX_: ',_NPRTMX_
      do PX = 1, _NPRTMX_
      do PY = 1, _NPRTMX_
         do ipsf = 001, 101
         do jpsf = 001, 101
            if (NIMu.le._NIMMXP_)
     .          psfpert(ipsf,jpsf,PX,PY,NIMu) = 0.00
            enddo
            enddo
         enddo
         enddo
 
      PSFFILE_USE = PSFFILE_INP
      GDCFILE_USE = GDCFILE_INP
      PSFFILE_LIB = 'NONE'
      GDCFILE_LIB = 'NONE'
 
      if (VERBOSE.ge.2) print*,'---> query_hdre: INSTRUM...'
      call query_hdre(FILENAME,'INSTRUME',STRING20_INSTRUME,-1)
      call query_hdre(FILENAME,'DETECTOR',STRING20_DETECTOR,-1)
 
      if (VERBOSE.ge.1) then
          print*,'              '
          print*,'QUERY HEADER: '
          print*,'    INSTRUME: ',STRING20_INSTRUME
          print*,'    DETECTOR: ',STRING20_DETECTOR
          print*,'              '
          endif
 
      INSTu  = -1
      NAXIS1 =  0
      NAXIS2 =  0
 
      if (index(FILENAME,'_drz.fits').ne.0.or.
     .    index(FILENAME,'_drc.fits').ne.0) then
          Ks = 1
          INSTu = 0
          NAXIS1 = i4_query_hdre(FILENAME,'NAXIS1  ',-1)
          NAXIS2 = i4_query_hdre(FILENAME,'NAXIS2  ',-1)
          allocate(pixn(NAXIS1,NAXIS2))
          allocate(pixo(NAXIS1,NAXIS2))
          allocate(pixc(NAXIS1,NAXIS2))
          allocate(pixp(NAXIS1,NAXIS2))
          allocate(pixx(NAXIS1,NAXIS2))
          allocate(pixq(NAXIS1,NAXIS2))
          allocate(pixw(NAXIS1,NAXIS2))
          print*,'CALL read_hstdrz_full...'
          call read_hstdrz_full(FILENAME,pixc, NAXIS1, NAXIS2,
     .                                   CRPIX1_INST, CRPIX2_INST,
     .                                   CRVAL1_INST, CRVAL2_INST,
     .                                   CD1_1_INST,  CD1_2_INST,
     .                                   CD2_1_INST,  CD2_2_INST,
     .                                   LOFLAG_INST, HIFLAG_INST,
     .                                   BDRY_XR,  BDRY_YR,
     .                                   FILT_INST,EXPT_INST,
     .                                   RDAT_INST,PROP_INST)
          print*,' RET read_hstdrz_full...'
          EXPT_EFF = 1.0
          GDCFILE_LIB = 'NONE'
          if (PSFFILE_INP(1:6).ne.'APPHOT') then
              RAP = 9.99
              SKI = 12
              SKO = 16
              DOAPPHOT = .true.
              endif
          xgc0 = CRPIX1_INST
          ygc0 = CRPIX2_INST
          do i = 0001, NAXIS1
          do j = 0001, NAXIS2
             pixq(i,j) = 0
             if (pixc(i,j).gt.HIFLAG_INST) pixq(i,j) = 1
             enddo
             enddo
          BDRY_XR(1,1) = 0001
          BDRY_YR(1,1) = 0001
          BDRY_XR(2,1) = NAXIS1
          BDRY_YR(2,1) = 0001
          BDRY_XR(3,1) = NAXIS1
          BDRY_YR(3,1) = NAXIS2
          BDRY_XR(4,1) = 0001
          BDRY_YR(4,1) = NAXIS2
          K = 1
          if (.true.) then
              print*, 'DRZ/DRC image!!  BDRY INFO...'
              print*,'-----1--> ',BDRY_XR(1,K),BDRY_YR(1,K)
              print*,'-----2--> ',BDRY_XR(2,K),BDRY_YR(2,K)
              print*,'-----3--> ',BDRY_XR(3,K),BDRY_YR(3,K)
              print*,'-----4--> ',BDRY_XR(4,K),BDRY_YR(4,K)
              endif
          goto 3
          endif
 
 
      if (STRING20_INSTRUME(2:6).eq.'WFPC2') then
          if (VERBOSE.gt.0) then
              write(*,'(''READ IN WFPC2...'',80a)') FILENAME
              endif
          Ks = 4
          INSTu = 1
          NAXIS1 = 1600
          NAXIS2 = 1600
          if (VERBOSE.ge.1) print*,'---> NAXIS: ',NAXIS2,NAXIS2
          allocate(pixn(NAXIS1,NAXIS2))
          allocate(pixo(NAXIS1,NAXIS2))
          allocate(pixc(NAXIS1,NAXIS2))
          allocate(pixp(NAXIS1,NAXIS2))
          allocate(pixx(NAXIS1,NAXIS2))
          allocate(pixq(NAXIS1,NAXIS2))
          call read_wfpc2_flt_full(FILENAME,pixc,
     .                             CRPIX1_INST, CRPIX2_INST,
     .                             CRVAL1_INST, CRVAL2_INST,
     .                             CD1_1_INST,  CD1_2_INST,
     .                             CD2_1_INST,  CD2_2_INST,
     .                             LOFLAG_INST, HIFLAG_INST,
     .                             BDRY_XR,  BDRY_YR,
     .                             FILT_INST,EXPT_INST,
     .                             RDAT_INST,PROP_INST)
          print*,'WFPC2 CRPIX1_INST: ',CRPIX1_INST
          print*,'WFPC2 CRPIX2_INST: ',CRPIX2_INST
          PSFFILE_LIB = '/grp/webpages/jayander/STDPSFs' //
     .                  '/WFPC2/PSFSTD_WFPC2_F555W.fits'
          GDCFILE_LIB = '/grp/webpages/jayander/STDGDCs' //
     .                  '/WFPC2/STDGDC_WFPC2_YR2000.fits'
          if (PSFFILE_INP(1:4).eq.'AUTO') PSFFILE_USE = PSFFILE_LIB
          if (GDCFILE_INP(1:4).eq.'AUTO') GDCFILE_USE = GDCFILE_LIB
          if (PSFFILE_INP(1:4).ne.'AUTO') PSFFILE_LIB = PSFFILE_INP
          if (GDCFILE_INP(1:4).ne.'AUTO') GDCFILE_LIB = GDCFILE_INP
          xgc0 = CRPIX1_INST
          ygc0 = CRPIX2_INST
          if (VERBOSE.ge.1) print*,'WFPC2        xgc0: ',xgc0,ygc0
          do i = 0001, NAXIS1
          do j = 0001, NAXIS2
             pixq(i,j) = 0
             if (pixc(i,j).gt.HIFLAG_INST) pixq(i,j) = 1
             enddo
             enddo
          write(66,*) 'WFPC2 CRPIX1_INST: ',CRPIX1_INST
          write(66,*) 'WFPC2 CRPIX2_INST: ',CRPIX2_INST
          write(66,*) 'WFPC2 CRVAL1_INST: ',CRVAL1_INST
          write(66,*) 'WFPC2 CRVAL2_INST: ',CRVAL2_INST
          call xryr2xcyc_stdgc(CRPIX1_INST,CRPIX2_INST,
     .                         xgc0,ygc0,GDCFILE_USE)
          write(66,*) '      GDCFILE_LIB: ',GDCFILE_LIB
          write(66,*) '      GDCFILE_INP: ',GDCFILE_INP
          write(66,*) '      GDCFILE_USE: ',GDCFILE_USE
          write(66,*) 'WFPC2        xgc0: ',xgc0
          write(66,*) 'WFPC2        ygc0: ',ygc0
          CRPIX1_INST = xgc0
          CRPIX2_INST = ygc0
          endif
 
 
      if (STRING20_INSTRUME(2:4).eq.'ACS'.and.
     .    STRING20_DETECTOR(2:4).eq.'HRC') then
          if (VERBOSE.gt.0) then
              write(*,'(''READ IN ACS/HRC IMAGE...'',80a)') FILENAME
              endif
          PSFFILE_LIB = '/grp/webpages/jayander/STDPSFs/ACSHRC/'
     .               // 'PSFSTD_ACSHRC_F606W.fits'
 
          GDCFILE_LIB = '/grp/webpages/jayander/STDGDCs/ACSHRC/'
     .               // 'STDGDC_ACSHRC_F606W.fits'
 
          if (PSFFILE_INP(1:4).eq.'AUTO') PSFFILE_USE = PSFFILE_LIB
          if (GDCFILE_INP(1:4).eq.'AUTO') GDCFILE_USE = GDCFILE_LIB
 
          Ks = 1
          INSTu = 7
          NAXIS1 = 1024
          NAXIS2 = 1024
          PMAX_PERT = 65000
          if (VERBOSE.ge.1) print*,'---> NAXIS: ',NAXIS2,NAXIS2
          allocate(pixn(NAXIS1,NAXIS2))
          allocate(pixo(NAXIS1,NAXIS2))
          allocate(pixc(NAXIS1,NAXIS2))
          allocate(pixp(NAXIS1,NAXIS2))
          allocate(pixx(NAXIS1,NAXIS2))
          allocate(pixq(NAXIS1,NAXIS2))
          call read_acshrc_flt_full(FILENAME,pixc,
     .                              CRPIX1_INST, CRPIX2_INST,
     .                              CRVAL1_INST, CRVAL2_INST,
     .                              CD1_1_INST,  CD1_2_INST,
     .                              CD2_1_INST,  CD2_2_INST,
     .                              LOFLAG_INST, HIFLAG_INST,
     .                              BDRY_XR,  BDRY_YR,
     .                              FILT_INST,EXPT_INST,
     .                              RDAT_INST,PROP_INST)
          print*,'ACS/HRC CRVAL1_INST: ',CRVAL1_INST,CRVAL2_INST
          print*,'ACS/HRC CRPIX1_INST: ',CRPIX1_INST,CRPIX2_INST
          print*,'ACS/HRC GDCFILE_USE: ',GDCFILE_USE
          call xryr2xcyc_stdgc(CRPIX1_INST,CRPIX2_INST,
     .                         xgc0,ygc0,GDCFILE_USE)
          print*,'ACS/HRC        xgc0: ',xgc0,ygc0
          do i = 0001, NAXIS1
          do j = 0001, NAXIS2
             pixq(i,j) = 0
             if (pixc(i,j).gt.HIFLAG_INST) pixq(i,j) = 1
             enddo
             enddo
          endif
 
 
      if (STRING20_INSTRUME(2:4).eq.'ACS'.and.
     .    STRING20_DETECTOR(2:4).eq.'WFC') then
          if (VERBOSE.gt.0) then
              write(*,'(''READ IN ACS/WFC IMAGE...'',80a)') FILENAME
              endif
          Ks = 2
          INSTu = 5
          NAXIS1 = 4096
          NAXIS2 = 4096
          PMAX_PERT = 55000
          if (VERBOSE.eq.1) print*,'---> NAXIS: ',NAXIS2,NAXIS2
          allocate(pixn(NAXIS1,NAXIS2))
          allocate(pixo(NAXIS1,NAXIS2))
          allocate(pixc(NAXIS1,NAXIS2))
          allocate(pixp(NAXIS1,NAXIS2))
          allocate(pixx(NAXIS1,NAXIS2))
          allocate(pixq(NAXIS1,NAXIS2))
          if (VERBOSE.eq.1) print*,'call read_acswfc_flt_full...'
          call read_acswfc_flt_full(FILENAME,pixc, pixq,
     .                              CRPIX1_INST, CRPIX2_INST,
     .                              CRVAL1_INST, CRVAL2_INST,
     .                              CD1_1_INST,  CD1_2_INST,
     .                              CD2_1_INST,  CD2_2_INST,
     .                              LOFLAG_INST, HIFLAG_INST,
     .                              BDRY_XR,  BDRY_YR,
     .                              FILT_INST,EXPT_INST,
     .                              RDAT_INST,PROP_INST)
          if (VERBOSE.ge.1) then
              print*,'ACS/WFC CRPIX1_INST: ',CRPIX1_INST
              print*,'ACS/WFC CRPIX2_INST: ',CRPIX2_INST
              endif
          PSFFILE_LIB = PSFFILE_USE
          if (PSFFILE_USE.eq.'AUTO') then
             PSFFILE_LIB = '/grp/webpages/jayander/STDPSFs/'
     .                  // 'ACSWFC/PSFSTD_ACSWFC_F606W.fits'
             if (FILT_INST(1:5).eq.'F435W') then
                 PSFFILE_LIB = '/grp/webpages/jayander/STDPSFs/'
     .                      // 'ACSWFC/PSFSTD_ACSWFC_F435W.fits'
                 endif
             if (FILT_INST(1:5).eq.'F475W') then
                 PSFFILE_LIB = '/grp/webpages/jayander/STDPSFs/'
     .                      // 'ACSWFC/PSFSTD_ACSWFC_F475W.fits'
                 endif
             if (FILT_INST(1:5).eq.'F606W') then
                 PSFFILE_LIB = '/grp/webpages/jayander/STDPSFs/'
     .                      // 'ACSWFC/PSFSTD_ACSWFC_F606W.fits'
                 endif
             if (FILT_INST(1:5).eq.'F775W') then
                 PSFFILE_LIB = '/grp/webpages/jayander/STDPSFs/'
     .                      // 'ACSWFC/PSFSTD_ACSWFC_F775W.fits'
                 endif
             if (FILT_INST(1:5).eq.'F814W') then
                 PSFFILE_LIB = '/grp/webpages/jayander/STDPSFs/'
     .                      // 'ACSWFC/PSFSTD_ACSWFC_F814W.fits'
                 endif
              PSFFILE_USE = PSFFILE_LIB
              endif
          GDCFILE_LIB = GDCFILE_USE
          if (GDCFILE_USE.eq.'AUTO') then
             GDCFILE_LIB = '/grp/webpages/jayander/STDGDCs/'
     .                  // 'ACSWFC/STDGDC_ACSWFC_F606W.fits'
             if (FILT_INST(1:5).eq.'F435W') then
                 GDCFILE_LIB = '/grp/webpages/jayander/STDGDCs/'
     .                      // 'ACSWFC/STDGDC_ACSWFC_F435W.fits'
                 endif
             if (FILT_INST(1:5).eq.'F475W') then
                 GDCFILE_LIB = '/grp/webpages/jayander/STDGDCs/'
     .                      // 'ACSWFC/STDGDC_ACSWFC_F475W.fits'
                 endif
             if (FILT_INST(1:5).eq.'F606W') then
                 GDCFILE_LIB = '/grp/webpages/jayander/STDGDCs/'
     .                      // 'ACSWFC/STDGDC_ACSWFC_F606W.fits'
                 endif
             if (FILT_INST(1:5).eq.'F775W') then
                 GDCFILE_LIB = '/grp/webpages/jayander/STDGDCs/'
     .                      // 'ACSWFC/STDGDC_ACSWFC_F775W.fits'
                 endif
             if (FILT_INST(1:5).eq.'F814W') then
                 GDCFILE_LIB = '/grp/webpages/jayander/STDGDCs/'
     .                      // 'ACSWFC/STDGDC_ACSWFC_F814W.fits'
                 endif
              GDCFILE_USE = GDCFILE_LIB
              endif
 
          if (PSFFILE_USE.eq.'NONE') stop 'PSFFILE_USE = NONE'
          if (GDCFILE_USE.eq.'NONE') stop 'PSFFILE_USE = NONE'
          call xryr2xcyc_stdgc(CRPIX1_INST,CRPIX2_INST,
     .                         xgc0,ygc0,GDCFILE_USE)
          print*,'ACS/WFC HIFLAG: ',HIFLAG
          if (HIFLAG.gt.0) then
              do i = 0001, NAXIS1
              do j = 0001, NAXIS2
                 pixq(i,j) = 0
                 if (pixc(i,j).gt.HIFLAG) pixq(i,j) = 1
                 enddo
                 enddo
              endif
          endif
 
 
      if (STRING20_INSTRUME(2:5).eq.'WFC3'.and.
     .    STRING20_DETECTOR(2:5).eq.'UVIS') then
          if (VERBOSE.gt.0) then
              write(*,'(''READ IN WFC3/UVIS IMAGE...'',80a)') FILENAME
              endif
          Ks = 2
          INSTu = 8
          NAXIS1 = 4096
          NAXIS2 = 4096
          PMAX_PERT = 65000
          if (VERBOSE.eq.1) print*,'---> NAXIS: ',NAXIS2,NAXIS2
          allocate(pixn(NAXIS1,NAXIS2))
          allocate(pixo(NAXIS1,NAXIS2))
          allocate(pixc(NAXIS1,NAXIS2))
          allocate(pixp(NAXIS1,NAXIS2))
          allocate(pixx(NAXIS1,NAXIS2))
          allocate(pixq(NAXIS1,NAXIS2))
          call read_wfc3uv_flt_full(FILENAME,pixc,pixq,
     .                              CRPIX1_INST, CRPIX2_INST,
     .                              CRVAL1_INST, CRVAL2_INST,
     .                              CD1_1_INST,  CD1_2_INST,
     .                              CD2_1_INST,  CD2_2_INST,
     .                              LOFLAG_INST, HIFLAG_INST,
     .                              BDRY_XR,  BDRY_YR,
     .                              FILT_INST,EXPT_INST,
     .                              RDAT_INST,PROP_INST)
          GDCFILE_LIB = '/grp/webpages/jayander/STDGDCs/'
     .               // 'WFC3UV/STDGDC_WFC3UV_F606W.fits'
          PSFFILE_LIB = '/grp/webpages/jayander/STDPSFs/'
     .               // 'WFC3UV/PSFSTD_WFC3UV_F606W.fits'
          if (FILT_INST(1:5).eq.'F438W') then
              PSFFILE_LIB = '/grp/webpages/jayander/STDPSFs/'
     .                   // 'WFC3UV/PSFSTD_WFC3UV_F438W.fits'
              GDCFILE_LIB = '/grp/webpages/jayander/STDGDCs/'
     .                   // 'WFC3UV/STDGDC_WFC3UV_F438W.fits'
              endif
          if (FILT_INST(1:5).eq.'F555W') then
              PSFFILE_LIB = '/grp/webpages/jayander/STDPSFs/'
     .                   // 'WFC3UV/PSFSTD_WFC3UV_F555W.fits'
              GDCFILE_LIB = '/grp/webpages/jayander/STDGDCs/'
     .                   // 'WFC3UV/STDGDC_WFC3UV_F555W.fits'
              endif
          if (FILT_INST(1:5).eq.'F606W') then
              PSFFILE_LIB = '/grp/webpages/jayander/STDPSFs/'
     .                   // 'WFC3UV/PSFSTD_WFC3UV_F606W.fits'
              GDCFILE_LIB = '/grp/webpages/jayander/STDGDCs/'
     .                   // 'WFC3UV/STDGDC_WFC3UV_F606W.fits'
              endif
          if (FILT_INST(1:5).eq.'F775W') then
              PSFFILE_LIB = '/grp/webpages/jayander/STDPSFs/'
     .                   // 'WFC3UV/PSFSTD_WFC3UV_F775W.fits'
              GDCFILE_LIB = '/grp/webpages/jayander/STDGDCs/'
     .                   // 'WFC3UV/STDGDC_WFC3UV_F775W.fits'
              endif
          if (FILT_INST(1:5).eq.'F814W') then
              PSFFILE_LIB = '/grp/webpages/jayander/STDPSFs/'
     .                   // 'WFC3UV/PSFSTD_WFC3UV_F814W.fits'
              GDCFILE_LIB = '/grp/webpages/jayander/STDGDCs/'
     .                   // 'WFC3UV/STDGDC_WFC3UV_F814W.fits'
              endif
          if (PSFFILE_INP(1:4).eq.'AUTO') PSFFILE_USE = PSFFILE_LIB
          if (GDCFILE_INP(1:4).eq.'AUTO') GDCFILE_USE = GDCFILE_LIB
          if (PSFFILE_INP(1:4).ne.'AUTO') PSFFILE_LIB = PSFFILE_USE
          if (PSFFILE_USE.eq.'NONE') stop 'PSFFILE_USE = NONE'
          if (GDCFILE_USE.eq.'NONE') stop 'PSFFILE_USE = NONE'
          if (VERBOSE.gt.0) then
              write(*,'(''WFC3UV CRPIX1_INST: '',f12.7)') CRPIX1_INST
              write(*,'(''WFC3UV CRPIX2_INST: '',f12.7)') CRPIX2_INST
              endif
          call xryr2xcyc_stdgc(CRPIX1_INST,CRPIX2_INST,
     .                         xgc0,ygc0,GDCFILE_USE)
          print*,'WFC3/UVIS HIFLAG: ',HIFLAG
          if (HIFLAG.gt.0) then
              do i = 0001, NAXIS1
              do j = 0001, NAXIS2
                 pixq(i,j) = 0
                 if (pixc(i,j).gt.HIFLAG) pixq(i,j) = 1
                 enddo
                 enddo
              endif
          endif
 
      if (STRING20_INSTRUME(2:5).eq.'WFC3'.and.
     .    STRING20_DETECTOR(2:3).eq.'IR') then
          if (VERBOSE.gt.0) then
              write(*,'(''READ IN WFC3IR IMAGE...'',80a)') FILENAME
              endif
          PSFFILE_LIB = '/grp/webpages/jayander/STDPSFs/WFC3IR/'
     .               // 'PSFSTD_WFC3IR_F105W.fits'
          GDCFILE_LIB = '/grp/webpages/jayander/STDGDCs/WFC3IR/'
     .               // 'STDGDC_WFC3IR.fits'
          if (PSFFILE_INP(1:4).eq.'AUTO') PSFFILE_USE = PSFFILE_LIB
          if (GDCFILE_INP(1:4).eq.'AUTO') GDCFILE_USE = GDCFILE_LIB
          Ks = 1
          INSTu  = 9
          NAXIS1 = 1014
          NAXIS2 = 1014
          if (VERBOSE.eq.1) print*,'---> NAXIS: ',NAXIS2,NAXIS2
          allocate(pixn(NAXIS1,NAXIS2))
          allocate(pixo(NAXIS1,NAXIS2))
          allocate(pixc(NAXIS1,NAXIS2))
          allocate(pixp(NAXIS1,NAXIS2))
          allocate(pixx(NAXIS1,NAXIS2))
          allocate(pixq(NAXIS1,NAXIS2))
          call read_wfc3ir_flt_full(FILENAME,pixc,
     .                              CRPIX1_INST, CRPIX2_INST,
     .                              CRVAL1_INST, CRVAL2_INST,
     .                              CD1_1_INST,  CD1_2_INST,
     .                              CD2_1_INST,  CD2_2_INST,
     .                              LOFLAG_INST, HIFLAG_INST,
     .                              BDRY_XR,  BDRY_YR,
     .                              FILT_INST,EXPT_INST,
     .                              RDAT_INST,PROP_INST)
          FMINu = FMIN/EXPT_INST
          PMAX_PERT = int(65000./EXPT_INST)
          PSFFILE_LIB = PSFFILE_USE
          if (PSFFILE_LIB(1:4).eq.'AUTO') then
             if (FILT_INST.eq.'F105W')
     .           PSFFILE_LIB = '/grp/webpages/jayander/STDPSFs/WFC3IR/'
     .                      // 'PSFSTD_WFC3IR_F105W.fits'
             if (FILT_INST.eq.'F110W')
     .           PSFFILE_LIB = '/grp/webpages/jayander/STDPSFs/WFC3IR/'
     .                      // 'PSFSTD_WFC3IR_F110W.fits'
             if (FILT_INST.eq.'F125W')
     .           PSFFILE_LIB = '/grp/webpages/jayander/STDPSFs/WFC3IR/'
     .                      // 'PSFSTD_WFC3IR_F125W.fits'
             if (FILT_INST.eq.'F140W')
     .           PSFFILE_LIB = '/grp/webpages/jayander/STDPSFs/WFC3IR/'
     .                      // 'PSFSTD_WFC3IR_F140W.fits'
             if (FILT_INST.eq.'F160W')
     .           PSFFILE_LIB = '/grp/webpages/jayander/STDPSFs/WFC3IR/'
     .                      // 'PSFSTD_WFC3IR_F160W.fits'
             endif
          if (VERBOSE.gt.0) then
              write(*,'(''WFC3/IR CRPIX1_INST: '',f12.7)') CRPIX1_INST
              write(*,'(''WFC3/IR CRPIX2_INST: '',f12.7)') CRPIX2_INST
              endif
          call xryr2xcyc_stdgc(CRPIX1_INST,CRPIX2_INST,
     .                         xgc0,ygc0,GDCFILE_USE)
          do i = 0001, NAXIS1
          do j = 0001, NAXIS2
             pixq(i,j) = 0
             if (pixc(i,j).gt.HIFLAG_INST) pixq(i,j) = 1
             enddo
             enddo
          endif
 
      if (STRING20_INSTRUME(2:7).eq.'NIRISS'.and.
     .    STRING20_DETECTOR(2:4).eq.'NIS') then
          if (VERBOSE.gt.0) then
              write(*,'(''READ IN NIRISS IMAGE...'',80a)') FILENAME
              endif
          GDCFILE_USE = 'NONE'
          Ks = 1
          INSTu  = 11
          NAXIS1 = 2048
          NAXIS2 = 2048
          if (VERBOSE.eq.1) print*,'---> NAXIS: ',NAXIS2,NAXIS2
          allocate(pixn(NAXIS1,NAXIS2))
          allocate(pixo(NAXIS1,NAXIS2))
          allocate(pixc(NAXIS1,NAXIS2))
          allocate(pixp(NAXIS1,NAXIS2))
          allocate(pixx(NAXIS1,NAXIS2))
          allocate(pixq(NAXIS1,NAXIS2))
          call readfits_r4e(FILENAME,pixc,2048,2048,1)
          call readfits_i4e(FILENAME,pixn,2048,2048,2)
          do i = 0001, NAXIS1
          do j = 0001, NAXIS2
             qqqq = pixn(i,j)/2
             qqqq = qqqq - 2*(qqqq/2)
             if (pixc(i,j).eq.0.and.qqqq.eq.1) pixc(i,j) = 65100
             enddo
             enddo
          do i = 0001, NAXIS1                   ! turn off the
          do j = 0001, NAXIS2                   ! saturation of the iso
             if (pixc(i,j).ge.65000) then       ! hot pix...
                 if (pixc(i+1,j  ).lt.100.or.
     .               pixc(i-1,j  ).lt.100.or.
     .               pixc(i  ,j-1).lt.100.or.
     .               pixc(i  ,j+1).lt.100.or.
     .               pixc(i+1,j-1).lt.100.or.
     .               pixc(i-1,j-1).lt.100.or.
     .               pixc(i+1,j+1).lt.100.or.
     .               pixc(i-1,j+1).lt.100) pixc(i,j) = 0.
                 endif
             enddo
             enddo
 
          CRPIX1_INST =   1024
          CRPIX2_INST =   1024
          CRVAL1_INST =   0.00
          CRVAL2_INST =   0.00
          CD1_1_INST  =  -1.00
          CD1_2_INST  =   0.00
          CD2_1_INST  =   0.00
          CD2_2_INST  =   1.00
          LOFLAG_INST =  -100.00
          HIFLAG_INST = 60000.0
          BDRY_XR(1,1) = 1
          BDRY_YR(1,1) = 1
          BDRY_XR(2,1) = NAXIS1
          BDRY_YR(2,1) = 1
          BDRY_XR(3,1) = NAXIS1
          BDRY_YR(3,1) = NAXIS2
          BDRY_XR(4,1) = 1
          BDRY_YR(4,1) = NAXIS2
          call query_hdre(FILENAME,'TIME-OBS ',TIMESTR,-1)
          call query_hdre(FILENAME,'FILTER   ',  FILT1,-1)
          FILT_INST = FILT1(1:5)
          call query_hdre_r4(FILENAME,'EFFEXPTM',EXPT_INST,-1)
          RDAT_INST = rDAT
          PMAX_PERT = 65000
          PSFFILE_LIB = PSFFILE_USE
          PROP_INST = 00000
          xgc0 = 0.00
          ygc0 = 0.00
          do i = 0001, NAXIS1
          do j = 0001, NAXIS2
             pixq(i,j) = 0
             if (pixc(i,j).gt.HIFLAG_INST) pixq(i,j) = 1
             enddo
             enddo
          endif
 
 
    3 continue
      print*,'-----1--> ',BDRY_XR(1,K),BDRY_YR(1,K)
      print*,'-----2--> ',BDRY_XR(2,K),BDRY_YR(2,K)
      print*,'-----3--> ',BDRY_XR(3,K),BDRY_YR(3,K)
      print*,'-----4--> ',BDRY_XR(4,K),BDRY_YR(4,K)
 
 
      if (INSTu.eq.-1) then
          NAXIS1 = 0
          NAXIS2 = 0
          NAXIS1 = i4_query_hdre(FILENAME,'NAXIS1  ',-1)
          NAXIS2 = i4_query_hdre(FILENAME,'NAXIS2  ',-1)
          if (NAXIS1.le.0.or.NAXIS2.le.0) then
              print*,' FILENAME: ',FILENAME(1:60)
              print*,'   PREFIX: ',PREFIX(1:60)
              print*,'   NAXIS1: ',NAXIS1
              print*,'   NAXIS2: ',NAXIS2
              stop '---> no way found to read in image...'
              endif
          INSTu = 0
          Ks = 1
          allocate(pixn(NAXIS1,NAXIS2))
          allocate(pixo(NAXIS1,NAXIS2))
          allocate(pixc(NAXIS1,NAXIS2))
          allocate(pixp(NAXIS1,NAXIS2))
          allocate(pixx(NAXIS1,NAXIS2))
          allocate(pixq(NAXIS1,NAXIS2))
          allocate(pixw(NAXIS1,NAXIS2))
          CRPIX1_INST = 0.0
          CRPIX2_INST = 0.0
          CRVAL1_INST = 0.0
          CRVAL2_INST = 0.0
          CD1_1_INST  = 0.0
          CD1_2_INST  = 0.0
          CD2_1_INST  = 0.0
          CD2_2_INST  = 0.0
          LOFLAG_INST = -100.
          HIFLAG_INST = 1e6
          FILT_INST   = 'NONE'
          EXPT_INST   = 1.00
          RDAT_INST   = 0.00
          PROP_INST   = 00000
          call readfits_r4(FILENAME,pixc, NAXIS1, NAXIS2)
          print*,'                NAXIS1: ',NAXIS1
          print*,'                NAXIS2: ',NAXIS2
          print*,'pix(NAXIS1/2,NAXIS2/2): ',pixc(NAXIS1/2,NAXIS2/2)
          BDRY_XR(1,1) = 1
          BDRY_YR(1,1) = 1
          BDRY_XR(2,1) = NAXIS1
          BDRY_YR(2,1) = 1
          BDRY_XR(3,1) = NAXIS1
          BDRY_YR(3,1) = NAXIS2
          BDRY_XR(4,1) = 1
          BDRY_YR(4,1) = NAXIS2
          do i = 0001, NAXIS1
          do j = 0001, NAXIS2
             pixq(i,j) = 0
             if (pixc(i,j).gt.HIFLAG_INST) pixq(i,j) = 1
             enddo
             enddo
          endif
 
      HIFLAG_COMMON = HIFLAG_INST
      LOFLAG_COMMON = LOFLAG_INST
 
      if (PSFFILE_INP(1:4).eq.'AUTO') PSFFILE_USE = PSFFILE_LIB
      if (GDCFILE_INP(1:4).eq.'AUTO') GDCFILE_USE = GDCFILE_LIB
 
      EXPT_EFF = EXPT_INST
      EXPTu    = EXPT_INST
      if (INSTu.eq.0) EXPT_EFF = 1.00
      if (INSTu.eq.9) EXPT_EFF = 1.00
 
 
      if (As.ne.0) then
         print*,' '
         print*,'INSERT ASs   As = ',As
         print*,'        EXPT_INST: ',EXPT_INST
         print*,'        EXPT_EFF : ',EXPT_EFF
         print*,'        EXPTu    : ',EXPTu
         print*,' '
c        write(FILEOUT,'(''STEP6C_INSERT_ASs.'',i4.4)') NIMu
c        open(71,file=FILEOUT,status='unknown')
         do A = 1, As
            xr = xinp_a(A)
            yr = yinp_a(A)
            mr = minp_a(A)
            fr = 10**(-mr/2.5)
            call locpsfij_stdpsf(int(xr+0.5),int(yr+0.5),
     .                           psfloc,PSFFILE_USE)
            ixr0 = min(max(int(xr+0.5),1),NAXIS1)
            iyr0 = min(max(int(yr+0.5),1),NAXIS2)
            pixc0 = pixc(ixr0,iyr0)
            faddt = 0.
            faddn = 0.
            do i = max(0001,int(xr+0.5)-12), min(NAXIS1,int(xr+0.5)+12)
            do j = max(0001,int(yr+0.5)-12), min(NAXIS2,int(yr+0.5)+12)
               dx = i-xr
               dy = j-yr
               fadd = fr*rpsf_phot(dx,dy,psfloc)
               fsig = sqrt(fadd)/sqrt(EXPT_EFF/EXPT_INST)
               fnoi = fsig*noise_sig()
               pixc(i,j) = pixc(i,j) + fadd + fnoi
               faddt = faddt + fadd
               faddn = faddn + fadd + fnoi
c              write(72,172) i,j,xr,yr,dx,dy,rpsf_phot(dx,dy,psfloc),
c    .                       fadd, fsig, fnoi, pixc(i,j), faddt, faddn,
c    .                       A
c 172          format(1x,i4,1x,i4,1x,f8.2,1x,f8.2,
c    .                3x,f8.4,1x,f8.4,1x,f9.6,
c    .                3x,f9.2,1x,f9.2,1x,f9.2,
c    .                3x,f9.1,1x,f9.1,1x,f9.1,1x,'A',i6.6)
               enddo
               enddo
            if (A.eq.A/1000*1000.and.VERBOSE.gt.0)
     .      write( *,171) xinp_A(A), yinp_A(A), minp_A(A),
     .                    xr, yr, fr,
     .                    pixc0, pixc(ixr0,iyr0),A
c           write(71,171) xinp_A(A), yinp_A(A), minp_A(A),
c    .                    xr, yr, fr,
c    .                    pixc0, pixc(ixr0,iyr0),A,
c    .                    faddt, faddn
  171       format(1x,f10.3,1x,f10.3,1x,f8.4,5x,
     .             5x,f10.3,1x,f10.3,1x,f9.1,3x,2f9.1,1x,'A',i6.6,
     .             5x,f10.1,1x,f10.1)
            enddo
c        close(71)
c        close(72)
         print*,' '
         print*,' '
         endif
 
      do i = 0001, NAXIS1
      do j = 0001, NAXIS2
         pixo(i,j) = pixc(i,j)
         pixn(i,j) = 0
         enddo
         enddo
 
      if (DOSATD.and.INSTu.ge.1) then
          if (HIFLAG.eq.0) then
              print*,'NSATGLOM...'
  433         NSATGLOM = 0
              do i = 0002, 4095
              do j = 0002, 4095
                 if (pixq(i,j).ne.0) then
                     do ii = i-1, i+1
                     do jj = j-1, j+1
                        if (pixq(ii,jj).eq.0.and.
     .                      pixc(ii,jj).gt.HIFLAG_INST) then
                            NSATGLOM = NSATGLOM + 1
                            pixq(ii,jj) = 2
                            endif
                        enddo
                        enddo
                     endif
                 enddo
                 enddo
              if (NSATGLOM.gt.0) goto 433
              endif
         call peak_sat_NAXIS(pixc,pixp,pixx,pixq,HIFLAG_INST,
     .                       NAXIS1,NAXIS2)
         call sat_phot_NAXIS(pixc,pixp,pixx,pixq,HIFLAG_INST,
     .                       NAXIS1,NAXIS2,PSFFILE_USE,
     .                       isat,jsat,zsat,NSATs)
c        call writfits_r4('pixo.fits',pixo,NAXIS1,NAXIS2)
c        call writfits_r4('pixc.fits',pixc,NAXIS1,NAXIS2)
c        call writfits_r4('pixp.fits',pixp,NAXIS1,NAXIS2)
c        call writfits_r4('pixx.fits',pixx,NAXIS1,NAXIS2)
c        call writfits_b1('pixq.fits',pixq,NAXIS1,NAXIS2)
c
c        print*,'                    '
c        print*,' pixc(2169,1999):    ',pixc(2169,1999)
c        print*,' pixq(2169,1999):    ',pixq(2169,1999)
c        print*,'                    '
c        print*,' pixc(2264,1171):    ',pixc(2264,1171)
c        print*,' pixq(2264,1171):    ',pixq(2264,1171)
c        print*,'                    '
c        if (.true.) STOP
c
c         do i = 0001, 4096
c         do j = 0001, 4096
c            if (pixo(i,j).gt.49000) then
c               write(44,144) i, j, pixo(i,j), pixq(i,j), pixc(i,j)
c  144           format(1x,i4,1x,i4,1x,f8.0,1x,i1,1x,f12.0)
c                endif
c             enddo
c             enddo
 
         endif
 
      if (VERBOSE.gt.0) then
          print*,'               '
          print*,'   INSTRUMENT: ',STRING20_INSTRUME
          print*,'     DETECTOR: ',STRING20_DETECTOR
          print*,'               '
          print*,'pixc(500,500): ',pixc(500,500)
          write(*,'('' CRPIX1_INST: '',2f12.7)') CRPIX1_INST,CRPIX2_INST
          write(*,'('' CRPIX1_xgc0: '',2f12.7)') xgc0,ygc0
          write(*,'('' CRVAL1_INST: '',2f12.7)') CRVAL1_INST,CRVAL2_INST
          write(*,'(''  CD1_1_INST: '',2f12.7)') CD1_1_INST,CD1_2_INST
          write(*,'(''  CD2_1_INST: '',2f12.7)') CD2_1_INST,CD2_2_INST
          write(*,'('' LOFLAG_INST: '',i8)') LOFLAG_INST
          write(*,'('' HIFLAG_INST: '',i8)') HIFLAG_INST
          write(*,'(''BDRY_XR(1,1): '',2f12.7)') BDRY_XR(1,1),
     .                                           BDRY_YR(1,1)
          write(*,'(''BDRY_XR(2,1): '',2f12.7)') BDRY_XR(2,1),
     .                                           BDRY_YR(2,1)
          write(*,'(''BDRY_XR(3,1): '',2f12.7)') BDRY_XR(3,1),
     .                                           BDRY_YR(3,1)
          write(*,'(''BDRY_XR(4,1): '',2f12.7)') BDRY_XR(4,1),
     .                                           BDRY_YR(4,1)
          write(*,'(''   FILT_INST: '',a5  )') FILT_INST(1:5)
          write(*,'(''   EXPT_INST: '',f9.2)') EXPT_INST
          write(*,'(''   RDAT_INST: '',f9.2)') RDAT_INST
          write(*,'(''   PROP_INST: '',i5)')   PROP_INST
          write(*,'(''    RA2SEXIG: '',f12.7,a10)') CRVAL1_INST,
     .                                     ra2sexig(CRVAL1_INST)
          write(*,'(''    DE2SEXIG: '',f12.7,a10)') CRVAL2_INST,
     .                                     de2sexig(CRVAL2_INST)
          print*,'               '
          endif
 
 
c the image has been read in!
 
      RCD1_1_INST =  CD2_2_INST/(CD1_1_INST*CD2_2_INST
     .                          -CD1_2_INST*CD2_1_INST)
      RCD1_2_INST = -CD1_2_INST/(CD1_1_INST*CD2_2_INST
     .                          -CD1_2_INST*CD2_1_INST)
      RCD2_1_INST = -CD2_1_INST/(CD1_1_INST*CD2_2_INST
     .                          -CD1_2_INST*CD2_1_INST)
      RCD2_2_INST =  CD1_1_INST/(CD1_1_INST*CD2_2_INST
     .                          -CD1_2_INST*CD2_1_INST)
 
      if (VERBOSE.ge.1) print*,'NSATs: ',NSATs
 
      if (SHOW_MSK.ne.'X') then
         allocate(pixm(NAXIS1,NAXIS2))
         do i = 001, NAXIS1
         do j = 001, NAXIS2
            pixm(i,j) = 0.
            enddo
            enddo
         call pix2mask(pixm,INSTu,
     .                 NSATs,isat,jsat,zsat,
     .                 NAXIS1,NAXIS2)
         if (SHOW_MSK(1:4).eq.'AUTO') then
            IMMSKu = PREFIX(1:9) // '_msk.fits'
            call writfits_r4(IMMSKu,pixm,NAXIS1,NAXIS2)
            endif
         endif
 
      if (VERBOSE.ge.2) then
         write(*,'(''#            '')')
         write(*,'(''#      NIMu: '',2i4)')  NIMu
         write(*,'(''#      NIMs: '',2i4)')  NIMs
         write(*,'(''#    NAXIS1: '',i4)')   NAXIS1
         write(*,'(''#    NAXIS2: '',i4)')   NAXIS2
         write(*,'(''#    NPERTs: '',i4)')   NPERTs
         write(*,'(''#FOCUS_LEVi: '',f9.4)') FOCUS_LEVELi
         write(*,'(''#  FILENAME: '',80a)')  FILENAME
         write(*,'(''#  FILTNAME: '',80a)')  FILTNAME
         write(*,'(''#       PRE: '',80a)')  PREFIX
         write(*,'(''#      STEM: '',80a)')  STEM
         write(*,'(''#      SUFF: '',80a)')  SUFFIX
         write(*,'(''# FILT_INST: '',80a)')  FILT_INST
         write(*,'(''# EXPT_INST: '',f9.4)') EXPT_INST
         write(*,'(''# EXPT_EFF : '',f9.4)') EXPT_EFF
         write(*,'(''# RDAT_INST: '',f9.4)') RDAT_INST
         write(*,'(''# PROP_INST: '',i5.5)') PROP_INST
         write(*,'(''#            '')')
         endif
 
 
 
c-------------------------------------------------
c
c find the reference frame
c
      if (VERBOSE.ge.2) print*,'WCSMODE: ',WCSMODE
      dRA = 0.
      dDE = 0.
      CRVAL1_OUTP = CRVAL1_INST
      CRVAL2_OUTP = CRVAL2_INST
      if (.not.(WCSMODE(1:4).eq.'NONE')) then
         if (VERBOSE.ge.2) then
            print*,'CALL FILL_WCS_INFO: '
            print*,'           WCSMODE: ',WCSMODE
            endif
         call fill_wcs_info(WCSMODE,OBJ_NAME,
     .                      CRPIX1_OUTP,CRPIX2_OUTP,
     .                      CRVAL1_OUTP,CRVAL2_OUTP,
     .                       CD1_1_OUTP, CD1_2_OUTP,
     .                       CD2_1_OUTP, CD2_2_OUTP,
     .                      RCD1_1_OUTP,RCD1_2_OUTP,
     .                      RCD2_1_OUTP,RCD2_2_OUTP)
         dRA = rd2x(CRVAL1_INST,CRVAL2_INST,CRVAL1_OUTP,CRVAL2_OUTP)
         dDE = rd2y(CRVAL1_INST,CRVAL2_INST,CRVAL1_OUTP,CRVAL2_OUTP)
         if (VERBOSE.ge.1) then
             write(*,'(''WCSMODE: '',a9,5x,a11,5x,2f10.5,5x,2f10.2)')
     .                  WCSMODE(1:9),OBJ_NAME,CRVAL1_OUTP,CRVAL2_OUTP,
     .                                        CRPIX1_OUTP,CRPIX2_OUTP
             endif
         if (VERBOSE.ge.2) then
            print*,'CALL FILL_WCS_INFO OBJ_NAME: ',OBJ_NAME
            print*,'CALL FILL_WCS_INFO   CRPIX1: ',CRPIX1_OUTP,
     .                                             CRPIX2_OUTP
            print*,'CALL FILL_WCS_INFO   CRVAL1: ',CRVAL1_OUTP,
     .                                             CRVAL2_OUTP
            endif
         endif
 
      PREFIX_LEN = -1
      do i = 1,76
         if (prefix(i:i).ne.' ') PREFIX_LEN = i
         enddo
 
      if (VERBOSE.ge.2) print*,' FIND_HISTMODE... '
      HMB = histmode_NAXIS(pixc,NAXIS1,NAXIS2)
      if (VERBOSE.gt.0) then
          write(*,'(''#  HISTMODE...  pixc (W/ SKYSUB): '',f8.2)') HMB
          endif
 
      PSFFILE_PRT = 'NONE'
 
      if (VERBOSE.ge.2) print*,'DOAPPHOT: ',DOAPPHOT
 
      if (DOAPPHOT) goto 333
 
      NAXIS_PSF = i4_query_hdre(PSFFILE_USE,'NAXIS   ',-1)
      if (NAXIS_PSF.eq.4.and.FOCUS_LEVELi.eq.0) then
         if (VERBOSE.ge.1) then
             print*,'MAKE ONE-FOCUS PSF FROM MEDIAN OF MULTI...'
             print*,'---> FOCUS_LEVELi: ',FOCUS_LEVELi
             print*,'--->    NAXIS_PSF: ',NAXIS_PSF
             endif
         PSFFILE_PFF = PREFIX(1:9) // '_plv.fits'
         FOCUS_LEVELu = 0.
         call loadstdpsf_stdpbf(FOCUS_LEVELu,PSFFILE_USE,PSFFILE_PFF)
         PSFFILE_USE = PSFFILE_PFF
         endif
 
      if (FOCUS_LEVELi.lt.0) then
         print*,' '
         print*,'CALL findfocus_stdpbf...  FOCUS_LEVELi = ',FOCUS_LEVELi
         print*,' '
         FOCUS_LEVELu = FOCUS_LEVELi
         NFOCu = 0
         call findfocus_stdpbf(pixc,HMB,HIFLAG_INST,NAXIS1,NAXIS2,
     .                         FOCUS_LEVELu,NFOCu,PSFFILE_USE,.false.)
         call loadstdpsf_stdpbf(FOCUS_LEVELu,PSFFILE_USE,PSFFILE_PFF)
         PSFFILE_PFF = PREFIX(1:9) // '_pff.fits'
         call savefits_stdpsf(PSFFILE_PFF)
         PSFFILE_USE = PSFFILE_PFF
         if (VERBOSE.ge.2) then
            print*,'call loadstdpsf_stdpbf: '
            print*,'--->  FOCUS_LEVi: ',FOCUS_LEVELi
            print*,'--->  FOCUS_LEVu: ',FOCUS_LEVELu
            print*,'---> PSFFILE_USE: ',PSFFILE_USE(1:40)
            print*,'---> PSFFILE_PFF: ',PSFFILE_PFF(1:40)
            print*,'--->       NFOCu: ',NFOCu
            endif
         endif
 
 
      if (VERBOSE.ge.2) then
          print*,'--->  FOCUS_LEVi: ',FOCUS_LEVELi
          print*,'--->  FOCUS_LEVu: ',FOCUS_LEVELu
          print*,'---> PSFFILE_LIB: ',PSFFILE_LIB(1:40)
          print*,'---> PSFFILE_USE: ',PSFFILE_USE(1:40)
          print*,'---> PSFFILE_PFF: ',PSFFILE_PFF(1:40)
          endif
 
      if (NPERTs.ne.0.and.NIMu.le._NIMMXP_) then
         print*,'                   '
         print*,'---> NPERTs: ',NPERTs
         print*,'                   '
         print*,'     call find_psfpert...'
         print*,'     PSFFILE_PFF = ',PSFFILE_PFF
         print*,'            NIMu = ',NIMu
         print*,'            PMAX = ',PMAX
         print*,'                   '
         call find_psfpert(pixc,HMB,PMAX_PERT,
     .                     NAXIS1,NAXIS2,
     .                     psfpert(1,1,1,1,NIMu),NPERTs,
     .                     PSFFILE_PFF)
         call output_psfperts(psfpert,NPERTs,STEMs_N,NIMs)
         PSFFILE_PRT = PREFIX(1:9) // '_psf.fits'
         print*,'                   '
         print*,'     PSFFILE_PRT = ',PSFFILE_PRT
         print*,'                 = ',psfpert(51,51,1,1,NIMu)
         print*,'                   '
         ptot = 0.
         do i = 01, NPERTs
         do j = 01, NPERTs
            do ipsf = 001, 101
            do jpsf = 001, 101
               ptot = ptot + abs(psfpert(ipsf,jpsf,i,j,NIMu))/
     .                                   NPERTs/NPERTs/16
               enddo
               enddo
            enddo
            enddo
         print*,'call psfpert2std: '
         print*,'--->        ptot: ',ptot
         print*,'---> PSFFILE_PRT: ',PSFFILE_PRT(1:40)
         call writfits_psfpert2std(psfpert(1,1,1,1,NIMu),
     .                             NPERTs,STEM,
     .                             NAXIS1,NAXIS2,
     .                             PSFFILE_PFF,
     .                             PSFFILE_PRT)
         endif
 
 
  333 continue ! do aperture photometry
 
      HIFLAG_COMMON = HIFLAG_INST
      LOFLAG_COMMON = LOFLAG_INST
 
      call query_hdre(FILENAME,'TIME-OBS ',TIMESTR,-1)
      call query_hdre(FILENAME,'DATE-OBS ',DATESTR,-1)
 
 
      write(HH(HHs+001),'(''# '')')
      write(HH(HHs+002),'(''#--------------------------------'')')
      write(HH(HHs+003),'(''# BASIC RUNTIME PARAMETERS          '')')
      write(HH(HHs+004),'(''#--------------------------------'')')
      write(HH(HHs+005),'(''#     HMIN: '',i09  )') HMIN
      write(HH(HHs+006),'(''#     FMIN: '',2f21.1)') FMIN,FMINu
      write(HH(HHs+007),'(''#     PMAX: '',f21.1)') PMAX
      write(HH(HHs+008),'(''# GDCFILEI: '',a80)')  GDCFILE_INP
      write(HH(HHs+009),'(''# GDCFILEL: '',a80)')  GDCFILE_LIB
      write(HH(HHs+010),'(''# GDCFILEU: '',a80)')  GDCFILE_USE
      write(HH(HHs+011),'(''# PSFFILEI: '',a80)')  PSFFILE_INP
      write(HH(HHs+012),'(''# PSFFILEL: '',a80)')  PSFFILE_LIB
      write(HH(HHs+013),'(''# PSFFILEU: '',a80)')  PSFFILE_USE
      write(HH(HHs+014),'(''# PSFFILEP: '',a80)')  PSFFILE_PRT
      if (NPERTs.ne.0) PSFFILE_USE = PSFFILE_PRT
      write(HH(HHs+015),'(''# PSFFILEU: '',a80)')  PSFFILE_USE
      write(HH(HHs+016),'(''#   NPERTs: '',i09)')  NPERTs
      write(HH(HHs+017),'(''# FOC_LEVi: '',f8.3)') FOCUS_LEVELi
      write(HH(HHs+018),'(''# FOC_LEVu: '',f8.3)') FOCUS_LEVELu
      write(HH(HHs+019),'(''# FILENAME: '',a80)')  FILENAME
      write(HH(HHs+020),'(''# HIFLAG  : '',i6)') HIFLAG
      write(HH(HHs+021),'(''# HIFLAG_I: '',i6)') HIFLAG_INST
      write(HH(HHs+022),'(''# LOFLAG_I: '',i6)') LOFLAG_INST
      write(HH(HHs+023),'(''# DOAPPHOT: '',l1)') DOAPPHOT
      write(HH(HHs+024),'(''# HISTMODE: '',f9.1)') HMB
      write(HH(HHs+025),'(''#      PID: '',i5.5)') PID
      write(HH(HHs+026),'(''#     FILT: '',80a )') FILTNAME
      write(HH(HHs+027),'(''#    EXPTI: '',f10.4)') EXPT_INST
      write(HH(HHs+028),'(''#    EXPTE: '',f10.4)') EXPT_EFF
      write(HH(HHs+029),'(''#     rDAT: '',f14.8)') rDATuu
      write(HH(HHs+030),'(''#  TIMESTR: '',20a)') TIMESTR
      write(HH(HHs+031),'(''#  DATESTR: '',20a)') DATESTR
      write(HH(HHs+032),'(''# FIND_OBJ: '',a11   )') OBJ_NAME
      write(HH(HHs+033),'(''#      rRA: '',4f15.8,'' ARCMIN'')')
     .                      rRA,RA0,dRA,dRA*60
      write(HH(HHs+034),'(''#      rDE: '',4f15.8,'' ARCMIN'')')
     .                      rDE,DE0,dDE,dDE*60
      write(HH(HHs+035),'(''# '')')
 
      HHs = HHs + 35
 
      if (.not.DOAPPHOT) then
         write(HH(HHs+001),'(''#--------------------------------'')')
         write(HH(HHs+002),'(''# PSF INFO '')')
         write(HH(HHs+003),'(''#--------------------------------'')')
         HHs = HHs + 003
         if (VERBOSE.ge.2) then
             print*,'  '
             print*,'HERE IS THE INFO ON THE PSF TO BE USED...'
             print*,'  '
             endif
         if (VERBOSE.ge.1) then
             print*,'PSFFILE_LIB: ',PSFFILE_LIB
             endif
         call infofits_stdpsf(PSFFILE_LIB,NXPSFs,NYPSFs,ilist,jlist)
         do i = 1, NXPSFs
         do j = 1, NYPSFs
            call locpsfij_stdpsf(ilist(i),jlist(j),
     .                           psfloc,PSFFILE_LIB)
            psfarr1(i,j) = psfloc(51,51)
            enddo
            enddo
         write(HH(HHs+1),'(''# CENTRAL PSFFILE_LIB: '',a80)')
     .                                 PSFFILE_LIB
         write(HH(HHs+2),'(''#   '',10x,10i8.4)')
     .                      (ilist(i),i=1,NXPSFs)
         write(HH(HHs+3),'(''#   '',10x,10i8.2)')
     .                      (      i ,i=1,NXPSFs)
         HHs = HHs + 3
         do j = NYPSFs,01,-1
            HHs = HHs + 1
            write(HH(HHs),'(''# '',4x,i4.4,1x,i2.2,1x,10f8.4)')
     .             jlist(j),j,(psfarr1(i,j),i=1,NXPSFs)
            enddo
         HHs = HHs + 1
         write(HH(HHs),'(''# '')')
         call infofits_stdpsf(PSFFILE_USE,NXPSFs,NYPSFs,ilist,jlist)
         do i = 1, NXPSFs
         do j = 1, NYPSFs
            call locpsfij_stdpsf(ilist(i),jlist(j),
     .                           psfloc,PSFFILE_USE)
            psfarr2(i,j) = psfloc(51,51)
            enddo
            enddo
         if (NPERTs.ne.0.and.NIMu.le._NIMMXP_) then
            write(HH(HHs+1),'(''# CENTRAL PERT PSF: '')')
            HHs = HHs + 1
            do j = NPERTs, 1, -1
               HHs = HHs + 1
               write(HH(HHs),'(''# '',4x,1x,i2.2,1x,10f8.4)')
     .               j,(psfpert(51,51,i,j,NIMu),i=1,NPERTs)
               enddo
            HHs = HHs + 1
            write(HH(HHs),'(''# '')')
            endif
         write(HH(HHs+1),'(''# CENTRAL PSFFILE_USE: '',a80)')
     .                                 PSFFILE_USE
         write(HH(HHs+2),'(''#   '',10x,10i8.4)')
     .                      (ilist(i),i=1,NXPSFs)
         write(HH(HHs+3),'(''#   '',10x,10i8.2)')
     .                      (      i ,i=1,NXPSFs)
         HHs = HHs + 3
         do j = NYPSFs,01,-1
            HHs = HHs + 1
            write(HH(HHs),'(''# '',4x,i4.4,1x,i2.2,1x,10f8.4)')
     .             jlist(j),j,(psfarr2(i,j),i=1,NXPSFs)
            enddo
         HHs = HHs + 1
         write(HH(HHs),'(''# '')')
         write(HH(HHs+1),'(''# CENTRAL  DIFFERENCe'')')
         write(HH(HHs+2),'(''#   '',10x,10i8.4)')
     .                      (ilist(i),i=1,NXPSFs)
         write(HH(HHs+3),'(''#   '',10x,10i8.2)')
     .                      (      i ,i=1,NXPSFs)
         HHs = HHs + 3
         do j = NYPSFs,01,-1
            HHs = HHs + 1
            write(HH(HHs),'(''# '',4x,i4.4,1x,i2.2,1x,10f8.4)')
     .             jlist(j),j,(psfarr2(i,j)-psfarr1(i,j),i=1,NXPSFs)
            enddo
         HHs = HHs + 1
         write(HH(HHs),'(''# '')')
         endif
 
      if (WCSMODE.ne.'NONE') then
         write(HH(HHs+01),'(''#---------------------------------'')')
         write(HH(HHs+02),'(''# WCSMODE: '',a80)') WCSMODE
         write(HH(HHs+03),'(''#---------------------------------'')')
         write(HH(HHs+04),'(''# GDC_FILE: '',a11)') GDCFILE_USE
         write(HH(HHs+05),'(''# FIND_OBJ: '',a11)') OBJ_NAME
         write(HH(HHs+06),'(''# CRVAL12o: '',2f14.8,2x,a10,2x,a12)')
     .                                     CRVAL1_OUTP, CRVAL2_OUTP,
     .                                     ra2sexig(CRVAL1_OUTP),
     .                                     de2sexig(CRVAL2_OUTP)
         write(HH(HHs+07),'(''# CRVAL12i: '',2f14.8,2x,a10,2x,a12)')
     .                                     CRVAL1_INST, CRVAL2_INST,
     .                                     ra2sexig(CRVAL1_INST),
     .                                     de2sexig(CRVAL2_INST)
         write(HH(HHs+08),'(''# CRVAL12d: '',2f14.8,2x,a8)')
     .             -60*rd2x(CRVAL1_INST,CRVAL2_INST,
     .                      CRVAL1_OUTP,CRVAL2_OUTP),
     .              60*rd2y(CRVAL1_INST,CRVAL2_INST,
     .                      CRVAL1_OUTP,CRVAL2_OUTP),
     .              '(arcmin)'
         write(HH(HHs+09),'(''# CRPIX1io: '',2f20.10)') CRPIX1_INST,
     .                                                  CRPIX1_OUTP
         write(HH(HHs+10),'(''# CRPIX2io: '',2f20.10)') CRPIX2_INST,
     .                                                  CRPIX2_OUTP
         write(HH(HHs+11),'(''# CRVAL1io: '',2f20.10)') CRVAL1_INST,
     .                                                  CRVAL1_OUTP
         write(HH(HHs+12),'(''# CRVAL2io: '',2f20.10)') CRVAL2_INST,
     .                                                  CRVAL2_OUTP
         write(HH(HHs+13),'(''#   CD1_1o: '',2f20.10)')  CD1_1_INST,
     .                                                   CD1_1_OUTP
         write(HH(HHs+14),'(''#   CD1_2o: '',2f20.10)')  CD1_2_INST,
     .                                                   CD1_2_OUTP
         write(HH(HHs+15),'(''#   CD2_1o: '',2f20.10)')  CD2_1_INST,
     .                                                   CD2_1_OUTP
         write(HH(HHs+16),'(''#   CD2_2o: '',2f20.10)')  CD2_2_INST,
     .                                                   CD2_2_OUTP
         write(HH(HHs+17),'(''#  RCD1_1o: '',2f20.10)') RCD1_1_INST,
     .                                                  RCD1_1_OUTP
         write(HH(HHs+18),'(''#  RCD1_2o: '',2f20.10)') RCD1_2_INST,
     .                                                  RCD1_2_OUTP
         write(HH(HHs+19),'(''#  RCD2_1o: '',2f20.10)') RCD2_1_INST,
     .                                                  RCD2_1_OUTP
         write(HH(HHs+20),'(''#  RCD2_2o: '',2f20.10)') RCD2_2_INST,
     .                                                  RCD2_2_OUTP
         HHs = HHs + 21
         endif
 
 
c-------------------------------------------------------
c
c now, go through and measure the saturated stars; this new
c version of it takes the PSF into account, and is most accurately
c done when PERT is used, since it allows account to be taken of
c how flux sloshes from the core to the outskirts, which without
c correction would result in a different zeropoint for saturated
c and unsaturated stars, since unsaturated stars use 5x5-pixel
c apertures, and saturated one generally use larger ones.
c
 
 
      if (VERBOSE.ge.2) then
         print*,'  '
         print*,'  '
         print*,'*************************************'
         print*,'*************************************'
         print*,'**   '
         print*,'**  GO THRU IMAGE AND FIND PEAKS... '
         print*,'**   '
         print*,'**     HMIN: ',HMIN
         print*,'**     FMIN: ',FMIN
         print*,'**    FMINu: ',FMINu
         print*,'**     PMAX: ',PMAX
         print*,'**     NIMu: ',NIMu
         print*,'**     NIMs: ',NIMs
         print*,'**   '
         print*,'** LOFLAG_I: ',LOFLAG_INST
         print*,'** HIFLAG_I: ',HIFLAG_INST
         print*,'** HIFLAG  : ',HIFLAG
         print*,'**   '
         print*,'**   NAXIS1: ',NAXIS1
         print*,'**   NAXIS2: ',NAXIS2
         print*,'**   '
         print*,'*************************************'
         print*,'*************************************'
         print*,'  '
         print*,'  '
         endif
 
      if (VERBOSE.ge.1) then
         write( *,378)
         write( *,278)
         write( *,378)
         endif
 
      do hu = 1, 16
         hhist(hu) = 0
         enddo
 
      if (VERBOSE.ge.2) print*,'SHOW_ALL: ',SHOW_ALL
      if (SHOW_ALL(1:3).eq.'YES') then
         if (VERBOSE.ge.2) then
             print*,'            '
             print*,'SHOW_ALL: ',SHOW_ALL(1:20)
             print*,'            '
             print*,'---> output lots of diagnostic stuff...'
             print*,'            '
             endif
         call writfits_r4('pixo.fits',pixo,NAXIS1,NAXIS2)
         call writfits_r4('pixc.fits',pixc,NAXIS1,NAXIS2)
         call writfits_r4('pixp.fits',pixp,NAXIS1,NAXIS2)
         call writfits_r4('pixx.fits',pixx,NAXIS1,NAXIS2)
         if (INSTu.eq.0) then
            call writfits_r4('pixw.fits',pixw,NAXIS1,NAXIS2)
            call writfits_b1('pixq.fits',pixq,NAXIS1,NAXIS2)
            endif
         endif
 
      if (VERBOSE.ge.2) print*,'SHOW_USE: ',SHOW_USE
      if (SHOW_USE(1:2).ne.'NO'.and.SHOW_USE.ne.'GCTEMP') then
          if (VERBOSE.ge.1) print*,'SHOW_USE...',SHOW_USE
          SHOW_USEu = SHOW_USE
          if (VERBOSE.ge.1) print*,'SHOW_USEu..',SHOW_USEu
          if (SHOW_USEu(1:4).eq.'AUTO'.or.
     .        SHOW_USEu(1:6).eq.'GCTEMP') then
             !SHOW_USEu = PREFIX(1:9) // '_org.fits'
             !call writfits_r4(SHOW_USEu,pixo,NAXIS1,NAXIS2)
              SHOW_USEu = PREFIX(1:9) // '_use.fits'
              endif
          call writfits_r4(SHOW_USEu,pixc,NAXIS1,NAXIS2)
          endif
 
      if (VERBOSE.ge.1) print*,'SHOW_FND: ',SHOW_FND
      if (SHOW_FND(1:2).ne.'NO') then
         allocate(pixy(NAXIS1,NAXIS2))
         do i = 0001, NAXIS1
         do j = 0001, NAXIS2
            pixy(i,j) = 0
            enddo
            enddo
         endif
 
      if (VERBOSE.ge.1) print*,'SHOW_SUB: ',SHOW_SUB
      if (SHOW_SUB(1:4).eq.'AUTO') then
         print*,'ALLOCATE SUB...',NAXIS1,NAXIS2
         allocate(pixs(NAXIS1,NAXIS2))
         do i = 001, NAXIS1
         do j = 001, NAXIS2
            pixs(i,j) = pixc(i,j)
            enddo
            enddo
         endif
 
 
 
c--------------------------------------
c go through the image pixel by pixel to
c see if each one qualifies as a source worth
c "finding" and measuring
c
      if (VERBOSE.ge.1) then
          print*,'                '
          print*,'ENTER iit,jjt...'
          print*,'        NAXIS1: ',NAXIS1
          print*,'        NAXIS2: ',NAXIS2
          print*,'      DOAPPHOT: ',DOAPPHOT
          print*,'                '
          endif
 
 
      Ns = 0
      Nx = 0 ! this will keep track of the total found
 
      do jjt = 5, NAXIS2-5
      do iit = 5, NAXIS1-5
 
         if (Ns.ge._NSTMAX_) goto 444
 
         SATD = .false.
 
         ii = iit
         jj = jjt
 
         hu  = 1
         kru = 0
         do K = 1, Ks
            if (inside_poly(iit*1.0d0,jjt*1.0d0,
     .                      BDRY_XR(1,K),BDRY_YR(1,K),4))
     .                      kru = K
            enddo
         if (kru.eq.0) goto 444
         if (KSEL.ne.0.and.kru.ne.KSEL) goto 444
 
         xr = 0.
         yr = 0.
         mr = 0.
         fr = 0.
         sr = 0.
         qr  = 0.00
         mmeth = 0
         irmin = 0
         irmax = 0
 
 
c------------------------
c 01) not a local max
c
         hu = 2
         do i = ii-1,ii+1
         do j = jj-1,jj+1
            if (pixc(i,j).gt.pixc(ii,jj)) goto 444
            enddo
            enddo
 
         neq = 0
         do i = ii-2,ii+2
         do j = jj-2,jj+2
            if (pixc(i,j).eq.pixc(ii,jj)) neq = neq + 1
            enddo
            enddo
         if (neq.gt.4) goto 444
 
 
 
c---------------------------------------
c 02) skip if too close to brighter pixel
c
         hu = 3
         hobs = fnd_hloc_NAXIS(ii,jj,pixc,NAXIS1,NAXIS2)
         if (hobs.lt.abs(HMIN)) goto 444
 
 
c-------------------------------------
c 03) skip if not enough flux in this peak
c
         hu = 4
         FEST = pixc(ii,jj) +
     .          max(pixc(ii+1,jj)+pixc(ii+1,jj+1)+pixc(ii,jj+1),
     .              pixc(ii+1,jj)+pixc(ii+1,jj-1)+pixc(ii,jj-1),
     .              pixc(ii-1,jj)+pixc(ii-1,jj+1)+pixc(ii,jj+1),
     .              pixc(ii-1,jj)+pixc(ii-1,jj-1)+pixc(ii,jj-1))
         minsky = min(
     .       (pixc(ii+1,jj+1)+pixc(ii  ,jj+1)+pixc(ii-1,jj+1))/03,
     .       (pixc(ii+1,jj-1)+pixc(ii  ,jj-1)+pixc(ii-1,jj-1))/03,
     .       (pixc(ii+1,jj+1)+pixc(ii+1,jj  )+pixc(ii+1,jj-1))/03,
     .       (pixc(ii-1,jj+1)+pixc(ii-1,jj  )+pixc(ii-1,jj-1))/03,
     .       (pixc(ii-2,jj-1)+pixc(ii-2,jj  )+pixc(ii-2,jj+1)+
     .        pixc(ii+2,jj-1)+pixc(ii+2,jj  )+pixc(ii+2,jj+1)+
     .        pixc(ii-1,jj-2)+pixc(ii  ,jj-2)+pixc(ii+1,jj-2)+
     .        pixc(ii-1,jj+2)+pixc(ii  ,jj+2)+pixc(ii+1,jj+2))/12)
 
         minsky =  mbar_sky_NAXIS(ii,jj,5,8,pixc,NAXIS1,NAXIS2)
 
         FESTR = FEST
         if (pixc(ii,jj).gt.HIFLAG.or.
     .       pixc(ii,jj).gt.HIFLAG_INST) minsky = HMB
         FEST  = FEST - 4*minsky
c        if (FEST.lt.FMIN*0.1) goto 444
         if (FEST.lt.FMINu) goto 444
 
 
c-----------------------------------------------
c 04) skip if this is below them mask prediction
c
         hu = 5
         if (SHOW_MSK(1:1).ne.'X') then
c            if (pixc(ii,jj).lt.HIFLAG.and.
c    .           FEST.lt.4*pixm(ii,jj)) goto 444
             if (pixc(ii,jj).lt.HIFLAG.and.
     .           FEST.lt.40*pixm(ii,jj)) goto 444
             endif
 
 
c----------------------------------------------------------
c 05) outside of the good bounds of the image
c
         hu = 6
         if (INSTu.eq.5.and.jj.ge.2046.and.jj.le.2050) goto 444
         if (INSTu.eq.8.and.jj.ge.2046.and.jj.le.2050) goto 444
 
c----------------------------------------------------------
c 06) loflag too close
c
         hu = 7
         do i = ii-3,ii+3
         do j = jj-3,jj+3
            if (pixc(i,j).le.LOFLAG_INST) goto 444
            enddo
            enddo
 
 
c---------------------------------------
c 07) skip above the saturation threshold, but we don't
c     want to do saturated stars
c
         hu = 8
         if (pixq(ii,jj).eq.1.and.(.not.DOSATD)) goto 444
         if (pixc(ii,jj).gt.PMAX.and.PMAX.lt.HIFLAG)  goto 444
 
 
 
c----------------------------------------------------------
c 08) skip saturated pixels that are not the max of a contiguious saturated
c     distribution
c
         hu = 9
         if (pixp(ii,jj).gt.0) then
             do i = -5, 5
             do j = -5, 5
                if (i**2+j**2.le.5.5**2) then
                   if (ii+i.ge.0001.and.ii+i.le.NAXIS1.and.
     .                 jj+j.ge.0001.and.jj+j.le.NAXIS2) then
                       if (pixp(ii,jj).lt.pixx(ii,jj)) goto 444
                       endif
                   endif
                enddo
                enddo
             endif
 
 
c--------------------------------------------------------------
c 09) skip if saturated and too close to bigger saturated pixels
c
         hu = 10
         if (pixq(ii,jj).eq.1) then
             do i = ii-2,ii+2
             do j = jj-3,jj+3
                if ((i.ne.ii.or.j.ne.jj).and.
     .              pixc(i,j).ge.pixc(ii,jj)) goto 444
                enddo
                enddo
             endif
 
 
c---------------------------------------------------
c 10) throw out if not saturated, but near weirdness
c
         hu = 11
         NSAT = 0
         do i = -3,+3
         do j = -3,+3
            if (i**2+j**2.le.3.5**2.and.
     .          pixq(ii+i,jj+j).eq.1) NSAT = NSAT + 1
            enddo
            enddo
         if (pixq(ii,jj).eq.0.and.NSAT.gt.0) goto 444
 
 
c-------------------------------------------------------------
c 11) found a peak worth measuring!
c     get centroid positions, 3x3 aperture flux (quick-n-easy)
c
         hu = 12
         sr = min(HMB,minsky)
         if (pixc(ii,jj).gt.HIFLAG) sr = HMB
         fxout = (pixc(ii+1,jj)-pixc(ii-1,jj))/2/
     .           (pixc(ii  ,jj)-min(pixc(ii+1,jj),pixc(ii-1,jj)))
         fyout = (pixc(ii,jj+1)-pixc(ii,jj-1))/2/
     .           (pixc(ii,jj  )-min(pixc(ii,jj+1),pixc(ii,jj-1)))
         fout  = pixc(ii+1,jj+1)+pixc(ii  ,jj+1)+pixc(ii-1,jj+1)+
     .           pixc(ii+1,jj  )+pixc(ii  ,jj  )+pixc(ii-1,jj  )+
     .           pixc(ii+1,jj-1)+pixc(ii  ,jj-1)+pixc(ii-1,jj-1)-
     .           9*sr
 
         if (fout.lt.1) fout = FEST
         if (.not.(fxout.lt. 1.0)) fxout = 0.
         if (.not.(fxout.gt.-1.0)) fxout = 0.
         if (.not.(fyout.lt. 1.0)) fyout = 0.
         if (.not.(fyout.gt.-1.0)) fyout = 0.
         if (.not.( fout.gt.-1.0))  fout = 1.
 
         if (pixq(ii,jj).eq.1) then  ! saturated pixel!!!
             hu = 14
             SATD = .true.
             xr = ii
             yr = jj
             fr = pixo(ii,jj)/psfloc(51,51)
             do i = -3, 3
             do j = -3, 3
                if (i**2+j**2.le.3.5**2) then
                    if (fr*psfloc(51+i*4,51+j*4)*0.1.gt.
     .                  pixo(ii+i,jj+j)-sr) then          ! saturated CR!
c                       write(56,156) ii, jj
c 156                   format(1x,i4,1x,i4,1x,8f12.1)
                        goto 444
                        endif
                    endif
                enddo
                enddo
             fr = pixc(ii,jj)
             hu = 16
             sr = mbar_sky_NAXIS(ii,jj,12,16,pixc,NAXIS1,NAXIS2)
             cr = 0.
             if (INSTu.eq.9) then ! WFC3/IR
                 call fitsat_ir(xr,yr,sr,fr,qr,psfloc,pixc,
     .                           NAXIS1,NAXIS2,LOFLAG_INST,HIFLAG_INST)
                 endif
             if (INSTu.eq.5.or.INSTu.eq.8) then ! ACS/WFC or WFC3/UV
                 xr4 = xr
                 yr4 = yr
                 sr4 = sr
                 fr4 = fr
                 call fitsat_ccdXthenY(psfloc,pixo,NAXIS1,NAXIS2,sr4,
     .                                 xr4,yr4,fr4,ysigfit)
                 xr = xr4
                 yr = yr4
                 endif
             mmeth = 9
             goto 137
             endif
 
         if (DOAPPHOT) then
             hu = 12
             if (RAP.eq.0) then  ! backdoor... easiest possible thing to do,
                fr = fout        ! measure the brightest 4/9 pixels...
                xr = ii + fxout
                yr = jj + fyout
                goto 137
                endif
             xr = ii + fxout
             yr = jj + fyout
             fr = apphot_NAXIS(ii,jj,RAP,SKI,SKO,pixc,ss,
     .                         NAXIS1,NAXIS2)
             sr = ss
             hu = 15
             goto 137
             endif
 
         xro = ii + fxout
         yro = jj + fyout
         fro = fout
 
         call locpsfij_stdpsf(ii,jj,psfloc,PSFFILE_USE)              ! get local PSF
 
         sr = mbar_skyopt_NAXIS(xro,yro,pixc,psfloc,NAXIS1,NAXIS2)   ! find PSF-corr sky
 
 
c-----------------------------------------------------
c find my best postion, then flux
c
         xr = xro
         yr = yro
         call find_xyzXX_NAXIS(xr,yr,fr,sr,qr,pixc,psfloc,
     .                         NAXIS1,NAXIS2,1) ! 1 means weight by S/N
 
         fr = z_xyoptXX_NAXIS(xr,yr,sr,pixc,2.5,psfloc,
     .                        NAXIS1,NAXIS2,1)  ! 1 means to weight by S/N
 
         if (.not.SATD.and.qr.gt.QMAX.and.(.not.QCAND)) goto 444
 
         hu = 13
 
         dx = ii-xr
         dy = jj-yr
         cr =(pixc(ii,jj) - sr - fr*rpsf_phot(dx,dy,psfloc))/fr
         if (.not.(cr.lt. 0.999)) cr =  0.999
         if (.not.(cr.gt.-0.999)) cr = -0.999
         if (.not.SATD) then
            if (.not.QCAND.or.qr.gt.QMAX) then
               if (cr.gt.CMAX) goto 444
               if (cr.lt.CMIN) goto 444
               endif
            endif
 
         hu = 15
 
         if (fr.lt.1) then  ! if the "best" flux is found
            xr = xro        ! to be negative, then go with something
            yr = yro        ! cruder; we know it's a peak, so it *should*
            fr = fro        ! have a positve flux
            sr = sro
            mmeth = 0
            endif
 
c
c this code is necessary because sometimes the
c saturation flags are not set for pixels that
c are actually saturated...
c
 
         NSATx = 0
         do i = ii-2, ii+2
         do j = jj-2, jj+2
            if (pixc(i,j).gt.HIFLAG_INST) NSATx = NSATx + 1
            enddo
            enddo
c        write(67,167) xr, yr, -2.5*log10(fr), fr, qr, NSATx,
c    .                 ii, jj, pixc(ii,jj)
c 167    format(1x,f8.2,1x,f8.2,1x,f8.3,1x,f11.1,1x,f8.4,
c    .          1x,3i5,1x,f8.1)
 
         if (pixc(ii,jj).gt.HIFLAG_INST) then
            if (NSATx.gt.2.or.
     .          pixc(ii,jj+1).gt.HIFLAG_INST.or.
     .          pixc(ii,jj-1).gt.HIFLAG_INST) then ! emergency protocol
                if (.not.DOSATD) then
                    hu = 8
                    goto 444
                    endif
                xtot = 0.
                ytot = 0.
                ptot = 0.
                ftot = 0.
                do i = ii-2, ii+2
                do j = jj-2, jj+2
                   xtot = xtot + (pixc(i,j)-sr)*i
                   ytot = ytot + (pixc(i,j)-sr)*j
                   ptot = ptot + (pixc(i,j)-sr)
                   ftot = ftot + rpsf_phot((i-ii)*1.0,(j-jj)*1.0,psfloc)
                   enddo
                   enddo
                fr = ptot/ftot
                xr = xtot/ptot
                yr = ytot/ptot
                qr = 0 ! international sign of saturation
                endif
            endif
c        write(68,167) xr, yr, -2.5*log10(fr), fr, qr, NSATx,
c    .                 ii, jj, pixc(ii,jj)
 
  137    continue
         if (.not.(fr.gt.1.00)) fr = 1.0
         if (.not.(fr.lt.1e19)) fr = 1.0
         mr = -2.5*log10(fr)
         call xryr2mc_stdgc(xr,yr,dmc,GDCFILE_USE)
         Ns = Ns + 1
         Nx = Nx + 1
         if (Ns.gt._NSTMAX_) goto 444
         if (Ns.gt._NSTMAX_) Ns = _NSTMAX_
         if ((VERBOSE.ge.9.and.SATD).or.
     .       (VERBOSE.ge.1.and.Ns.lt.010).or.
     .       (VERBOSE.ge.1.and.Ns.eq.Ns/00010*00010.and.Ns.lt.00100).or.
     .       (VERBOSE.ge.1.and.Ns.eq.Ns/00050*00050.and.Ns.lt.01000).or.
     .       (VERBOSE.ge.1.and.Ns.eq.Ns/00100*00100))
     .       write( *,178) Ns,xr,yr,mr,sr,mmeth,irmin,irmax,
     .                     int(pixc(ii  ,jj  )),
     .                     int(pixc(ii+1,jj  )),
     .                     int(pixc(ii-1,jj  )),
     .                     int(pixc(ii  ,jj+1)),
     .                     int(pixc(ii  ,jj-1)),
     .                     int(pixc(ii+1,jj+1)),
     .                     int(pixc(ii-1,jj+1)),
     .                     int(pixc(ii+1,jj-1)),
     .                     int(pixc(ii-1,jj-1)),
     .                     int(sr),int(FESTR),int(FEST)
 
         xgc = xr
         ygc = yr
         mgc = mr + dmc
 
         call xryr2xcyc_stdgc(xr,yr,xgc,ygc,GDCFILE_USE)
         dRA = (xgc-xgc0)*CD1_1_INST + (ygc-ygc0)*CD1_2_INST
         dDE = (xgc-xgc0)*CD2_1_INST + (ygc-ygc0)*CD2_2_INST
         RA = xy2r(dRA,dDE,CRVAL1_INST,CRVAL2_INST)
         DE = xy2d(dRA,dDE,CRVAL1_INST,CRVAL2_INST)
         kr = kru
 
         m1k = mgc + 2.5*log10(EXPT_EFF/1000.0)
         mzp = m1k
 
         ug = xgc
         vg = ygc
         if (WCSMODE.ne.'NONE') then
            dRA = rd2x(RA,DE,CRVAL1_OUTP,CRVAL2_OUTP)
            dDE = rd2y(RA,DE,CRVAL1_OUTP,CRVAL2_OUTP)
            ug = CRPIX1_OUTP + RCD1_1_OUTP*dRA + RCD1_2_OUTP*dDE
            vg = CRPIX2_OUTP + RCD2_1_OUTP*dRA + RCD2_2_OUTP*dDE
            endif
 
  178    format(i7,1x,f9.3,1x,f9.3,1x,f7.3,1x,f6.1,'|',i1,i2,i2,
     .                   '|',i7,'|',4i6,'|',4i6,'|',3i8,1x,1l1)
  278    format('     N ',1x,    ! i7
     .          '   XCEN  ',1x,  ! f9.3
     .          '   YCEN  ',1x,  ! f9.3
     .          '  MAG  ',1x,    ! f7.3
     .          '  SKY ' ,'|', ! f6.1
     .          'MiSoS'  ,'|', ! i1,1x,i1,1x,i2
     .          '   PCEN','|',   ! i7
     .          '   P02',        ! i6
     .          '   P03',
     .          '   P04',
     .          '   P05','|',
     .          '   P06',
     .          '   P07',
     .          '   P08',
     .          '   P09','|',
     .          '    mSKY',
     .          '    F2x2',
     .          ' F2x2-SK')
  378    format('-------',1x,'--------',1x,'--------',1x,
     .          '-------',1x,'------','|','- - -','|',
     .          '-------','|',
     .          ' ----- ----- ----- -----','|',
     .          ' ----- ----- ----- -----','|',
     .          ' ------- ------- -------')
 
 
          u_n(Ns) = ug
          v_n(Ns) = vg
         uu_n(Ns) = ug  ! may end up being corrected
         vv_n(Ns) = vg  ! by catalog
          m_n(Ns) = mr
          x_n(Ns) = xr
          y_n(Ns) = yr
          s_n(Ns) = sr
          k_n(Ns) = kr
         xx_n(Ns) = xgc
         yy_n(Ns) = ygc
         mm_n(Ns) = m1k ! pixel-area corrected
          r_n(Ns) = ra
          d_n(Ns) = de
          h_n(Ns) = hobs
         hh_n(Ns) = hobs
          i_n(Ns) = ii
          j_n(Ns) = jj
          p_n(Ns) = pixc(ii,jj)
          if (.not.(qr.lt.9.99)) qr = 9.99
          q_n(Ns) = qr
         sat_n(Ns) = 0
         if (SATD) sat_n(Ns) = 1
 
         c_n(Ns) = cr
 
         if (SHOW_SUB(1:4).eq.'AUTO') then
            do i = max(0001,ii-20), min(NAXIS1,ii+20)
            do j = max(0001,jj-20), min(NAXIS2,jj+20)
               dx = i-xr
               dy = j-yr
               pixs(i,j) = pixs(i,j) - fr*rpsf_phot(dx,dy,psfloc)
               enddo
               enddo
            endif
 
         pixc(ii,jj) = pixc(ii,jj)+1
         pixn(ii,jj) = Ns
 444     continue
         hhist(hu) = hhist(hu) + 1
         if (SHOW_FND(1:2).ne.'NO') pixy(ii,jj) = hu
 123     format(f9.3,1x,f9.3,1x,f8.4,1x,f9.1,1x,f8.2,5x,
     .          f9.3,1x,f9.3,1x,f8.4,1x,f9.1,1x,f8.2,1x,i6,1x,f6.2)
         enddo
         enddo
      if (VERBOSE.ge.1) then
          write( *,378)
          write( *,278)
          write( *,378)
          endif
 
      if (Ns.ne.Nx) then
         print*,'                                       '
         print*,'hst2xym problem: too many stars found: '
         print*,'                                       '
         print*,'      _NSTMAX_ = ',_NSTMAX_
         print*,'         Nx = ',Nx
         print*,'                                       '
         print*,'    need to increase _NSTMAX_          '
         print*,'                                       '
         stop
         endif
 
      if (SHOW_SUB(1:4).eq.'AUTO') then
          IMSUBu = PREFIX(1:9) // '_sub.fits'
          print*,'OUTPUT pix_sub: ',IMSUBu
          call writfits_r4(IMSUBu,pixs,NAXIS1,NAXIS2)
          if (SHOW_ALL(1:3).eq.'YES')
     .    call writfits_r4('pixs.fits',pixs,NAXIS1,NAXIS2)
          deallocate(pixs)
          endif
 
 
      if (SHOW_FND(1:2).ne.'NO') then
          SHOW_FNDu = SHOW_FND
          if (SHOW_FNDu(1:4).eq.'AUTO')
     .        SHOW_FNDu = PREFIX(1:9) // '_fnd.fits'
          print*,'OUTPUT pix_fnd: ',SHOW_FNDu
          call writfits_b1(SHOW_FNDu,pixy,NAXIS1,NAXIS2)
          if (SHOW_ALL(1:3).eq.'YES')
     .    call writfits_b1('pixy.fits',pixy,NAXIS1,NAXIS2)
          deallocate(pixy)
          endif
 
c----------------------------------------
c
c report on why some pixels weren't identified
c as stars
c
      write(HH(HHs+01),'(''#                                   '')')
      write(HH(HHs+02),'(''#-----------------------------------'')')
      write(HH(HHs+03),'(''# FINDING STATISTICS                '')')
      write(HH(HHs+04),'(''#-----------------------------------'')')
      HHs = HHs + 4
      if (VERBOSE.ge.1) print*,' '
      do hu = 1, 16
         if (VERBOSE.ge.1)
     .   write( *,277) hu,hhist(hu),hhist(hu)/40.96/4096,hhwhy(hu)
         write(HH(HHs+1),277) hu,hhist(hu),hhist(hu)*100./NAXIS1/NAXIS2,
     .                                     hhwhy(hu)
 277     format('#  ',i2,1x,i8,1x,f8.3,'%',3x,a20)
         HHs = HHs + 1
         enddo
      HHs = HHs + 1
      write(HH(HHs+1),'(''#'')')
      if (VERBOSE.ge.1) print*,' '
 
 
      if (VERBOSE.ge.2) then
         write(*,239)
         write(*,239) ' CRVAL_INST: ',CRVAL1_INST,CRVAL2_INST
         write(*,239) ' CD1_1_OUTP: ', CD1_1_INST, CD1_2_INST
         write(*,239) ' CD2_1_OUTP: ', CD2_1_INST, CD2_2_INST
         write(*,239) 'RCD1_1_OUTP: ',RCD1_1_INST,RCD1_2_INST
         write(*,239) 'RCD2_1_OUTP: ',RCD2_1_INST,RCD2_2_INST
         write(*,239) '            '
         write(*,239) 'CRVAL1_OUTP: ',CRVAL1_OUTP,CRVAL2_OUTP
         write(*,239) 'CRPIX1_OUTP: ',CRPIX1_OUTP,CRPIX2_OUTP
         write(*,239) ' CD1_1_OUTP: ', CD1_1_OUTP, CD1_2_OUTP
         write(*,239) ' CD2_1_OUTP: ', CD2_1_OUTP, CD2_2_OUTP
         write(*,239) 'RCD1_1_OUTP: ',RCD1_1_OUTP,RCD1_2_OUTP
         write(*,239) 'RCD2_1_OUTP: ',RCD2_1_OUTP,RCD2_2_OUTP
         write(*,239) '            '
  239    format(a13,3x,f15.8,3x,f15.8)
         endif
 
 
 
c---------------------------------------------------------------
c
c define the boundaries...
c
      if (VERBOSE.ge.2) then
         print*,'                '
         print*,'FIND BOUNDARY...'
         print*,'                '
         write(*,'(''# '')')
         write(*,442)
         write(*,'(''# '')')
         endif
c     write(HH(HHs+01),'(''# '')')
c     write(HH(HHs+02),'(''#-----------------------------------'')')
c     write(HH(HHs+03),442)
c     write(HH(HHs+04),'(''#-----------------------------------'')')
c     HHs = HHs + 4
      do K = 1, Ks
         do P = 1, 4
            xgc = 0.
            ygc = 0.
            RA = 0.
            DE = 0.
            xr = BDRY_XR(P,K)
            yr = BDRY_YR(P,K)
            call xryr2xcyc_stdgc(xr,yr,xgc,ygc,GDCFILE_USE)
            dRA = (xgc-xgc0)*CD1_1_INST + (ygc-ygc0)*CD1_2_INST
            dDE = (xgc-xgc0)*CD2_1_INST + (ygc-ygc0)*CD2_2_INST
            RA = xy2r(dRA,dDE,CRVAL1_INST,CRVAL2_INST)
            DE = xy2d(dRA,dDE,CRVAL1_INST,CRVAL2_INST)
            dRA = rd2x(RA,DE,CRVAL1_OUTP,CRVAL2_OUTP)
            dDE = rd2y(RA,DE,CRVAL1_OUTP,CRVAL2_OUTP)
            ug = xgc
            vg = ygc
            if (WCSMODE.ne.'NONE') then
               dRA = rd2x(RA,DE,CRVAL1_OUTP,CRVAL2_OUTP)
               dDE = rd2y(RA,DE,CRVAL1_OUTP,CRVAL2_OUTP)
               ug = CRPIX1_OUTP + RCD1_1_OUTP*dRA + RCD1_2_OUTP*dDE
               vg = CRPIX2_OUTP + RCD2_1_OUTP*dRA + RCD2_2_OUTP*dDE
               endif
            uu = ug
            vv = vg
            BDRY_XC(P,K) = xgc
            BDRY_YC(P,K) = ygc
            BDRY_UG(P,K) = ug
            BDRY_VG(P,K) = vg
            BDRY_RA(P,K) = RA
            BDRY_DE(P,K) = DE
            BDRY_UU(P,K) = uu
            BDRY_VV(P,K) = vv
c           write(HH(HHs+1),443) K,P,BDRY_XR(P,K),BDRY_YR(P,K),
c    .                               BDRY_XC(P,K),BDRY_YC(P,K),
c    .                               BDRY_UG(P,K),BDRY_VG(P,K),
c    .                               BDRY_RA(P,K),BDRY_DE(P,K),
c    .                               BDRY_UU(P,K),BDRY_VV(P,K)
c           HHs = HHs + 1
            if (VERBOSE.ge.2) then
                write(*,443) K,P,BDRY_XR(P,K),BDRY_YR(P,K),
     .                           BDRY_XC(P,K),BDRY_YC(P,K),
     .                           BDRY_UG(P,K),BDRY_VG(P,K),
     .                           BDRY_RA(P,K),BDRY_DE(P,K),
     .                           BDRY_UU(P,K),BDRY_VV(P,K)
                if (P.eq.4) write(*,'(''# '')')
                endif
            enddo
c        HHs = HHs + 1
         enddo
 
  443 format('# BDY_K',i1.1,'_P',i1.1,': ',
     .                     1x,f11.3,1x,f11.3,
     .                     1x,f11.3,1x,f11.3,
     .                     5x,f11.3,1x,f11.3,
     .                     5x,f15.8,1x,f15.8,
     .                     5x,f11.3,1x,f11.3)
  442 format('# CHIP_K  P_CRNR ',
     .       1x,'   XRAW        YRAW    ',
     .       1x,'    XGC         YGC    ',
     .       5x,'    UGC         VGC        ',
     .       5x,'     RA             Dec    ',
     .       5x,'     UU          VV        ')
 
 
 
c---------------------------------------------------------------
c
c find hh: the closest star in the list to the each star
c
      if (VERBOSE.ge.2) print*,'FIND CLOSEST TO TARGET... Ns: ',Ns
      do N = 1, Ns
         dm = 99.5
         do i = -9, 9
         do j = -9, 9
            dd = sqrt(1.*(i**2+j**2))
            if (dd.le.9.5.and.dd.lt.dm) then
               ii = i+i_n(N)
               jj = j+j_n(N)
               if (ii.ge.1.and.ii.le.NAXIS1.and.
     .             jj.ge.1.and.jj.le.NAXIS2) then
                      NN = pixn(ii,jj)
                      if (NN.ne.0.and.NN.ne.N) then
                         if (m_n(NN).lt.m_n(N)) dm=dd
                         endif
                   else
                      dm = dd
                   endif
               endif
            enddo
            enddo
         if (dm.gt.10) then
            do i = -99, 99
            do j = -99, 99
               dd = sqrt(1.*(i**2+j**2))
               if (dd.le.99.5.and.dd.lt.dm) then
                  ii = i+i_n(N)
                  jj = j+j_n(N)
                  if (ii.ge.1.and.ii.le.NAXIS1.and.
     .                jj.ge.1.and.jj.le.NAXIS2) then
                         NN = pixn(ii,jj)
                         if (NN.ne.0.and.NN.ne.N) then
                            if (m_n(NN).lt.m_n(N)) dm=dd
                            endif
                      else
                         dm = dd
                      endif
                  endif
               enddo
               enddo
            endif
         hh_n(N) = dm
         enddo!N
 
 
 
c---------------------------------------------------------------
c
c need to sort the stars by magnitude
c
      if (VERBOSE.ge.1) then
          print*,'          '
          print*,'DOMSORT...'
          endif
      do N = 1, Ns
         om_n(N) = N
         su_n(N) = m_n(N)
         enddo
      call qsorti(om_n,Ns,su_n)
      if (VERBOSE.ge.2) then
         print*,'---->   1: ',m_n(od_n(01))
         print*,'---->  Ns: ',m_n(od_n(Ns))
         endif
 
c---------------------------------------------------------------
c
c sort the stars by proximity (if desired)
c
      if (DOHSORT) then
         print*,' '
         print*,'DOHSORT...'
         print*,' '
         do N = 1, Ns
            od_n(N) = N
            su_n(N) = hh_n(N)
            enddo
         call qsorti(od_n,Ns,su_n)
         print*,'---->   1: ',hh_n(od_n(01))
         print*,'---->  Ns: ',hh_n(od_n(Ns))
         endif
 
 
 
c---------------------------------------------------------------
c
c find catalog...
c
      NMATs = 0
      AG  = 1.0
      BG  = 0.0
      CG  = 0.0
      DG  = 1.0
      u1o = 0.0
      v1o = 0.0
      u2o = 0.0
      v2o = 0.0
      if (CATMAT(1:5).eq.'2MASS') then
         print*,' '
         print*,'FIND 2MASS CATALOG... '
         stop 'ready for primetime?'
         print*,' '
         RA_MIN = 999.9
         RA_MAX = -99.9
         DE_MIN = 999.9
         DE_MAX = -99.9
         do k = 1, Ks
         do p = 1, 4
            if (BDRY_RA(P,K).lt.RA_MIN) RA_MIN = BDRY_RA(P,K)
            if (BDRY_RA(P,K).gt.RA_MAX) RA_MAX = BDRY_RA(P,K)
            if (BDRY_DE(P,K).lt.DE_MIN) DE_MIN = BDRY_DE(P,K)
            if (BDRY_DE(P,K).gt.DE_MAX) DE_MAX = BDRY_DE(P,K)
            enddo
            enddo
         write(*,'('' '')')
         write(*,'(''   RA_MIN: '',f12.7)') RA_MIN
         write(*,'(''   RA_MAX: '',f12.7)') RA_MAX
         write(*,'('' '')')
         write(*,'(''   DE_MIN: '',f12.7)') DE_MIN
         write(*,'(''   DE_MAX: '',f12.7)') DE_MAX
         write(*,'('' '')')
         call query_loc_2MASS(RA_MIN,RA_MAX,DE_MIN,DE_MAX,
     .                        ra_o,de_o,mj_o,mh_o,mk_o,aa_o,Os)
         print*,'---> Os: ',Os,Ks
         Us = 0
         do O = 1, Os
            dRA = rd2x(ra_o(O),de_o(O),CRVAL1_OUTP,CRVAL2_OUTP)
            dDE = rd2y(ra_o(O),de_o(O),CRVAL1_OUTP,CRVAL2_OUTP)
            ug_o(O) = CRPIX1_OUTP + RCD1_1_OUTP*dRA + RCD1_2_OUTP*dDE
            vg_o(O) = CRPIX2_OUTP + RCD2_1_OUTP*dRA + RCD2_2_OUTP*dDE
            kk_o(O) = 0
            do k = 1, Ks
                if (inside_poly(ug_o(O),vg_o(O),
     .                          BDRY_UG(1,K),BDRY_VG(1,K),4))
     .                          kk_o(O) = k
                enddo
            if (kk_o(O).ne.0) then
               Us = Us + 1
               ug_u(Us) = ug_o(O)
               vg_u(Us) = vg_o(O)
               mj_u(Us) = mj_o(O)
               oo_u(Us) = O
               endif
            enddo
         close(82)
         MSORTs = min(2*Us,Ns)
         do MSORT = 1, MSORTs
            N = om_n(MSORT)
            u_msort(MSORT) = u_n(N)
            v_msort(MSORT) = v_n(N)
            m_msort(MSORT) = m_n(N)
            n_msort(MSORT) = N
            dRA = rd2x(r_n(N),d_n(N),CRVAL1_OUTP,CRVAL2_OUTP)
            dDE = rd2y(r_n(N),d_n(N),CRVAL1_OUTP,CRVAL2_OUTP)
            ug = CRPIX1_OUTP + RCD1_1_OUTP*dRA + RCD1_2_OUTP*dDE
            vg = CRPIX2_OUTP + RCD2_1_OUTP*dRA + RCD2_2_OUTP*dDE
            enddo
c        close(81)
         print*,'---> MSORTs: ',MSORTs
         print*,'--->     Os: ',MSORTs
         print*,' '
         print*,'call find_offset_2MASS...'
         print*,' '
         dmax = 12.50
         call find_offset_2MASS(u_msort,v_msort,m_msort,MSORTs,
     .                          ug_u,   vg_u,   mj_u,   Us,
     .                          AG,BG,CG,DG,u1o,v1o,u2o,v2o,Lu,Ls,
     .                          dbar,dmax,
     .                          u1mat,v1mat,m1mat,n1mat,
     .                          u2mat,v2mat,m2mat,n2mat,NMATs)
         if (Lu.lt.5.or.Lu.lt.0.1*Ls) then               ! if can't find any 2MASS matches
             write(*,'(''# 2MASS FAILURE: '',2i8)') Lu,Ls
             print*,'---> Lu: ',Lu
             print*,'---> Ls: ',Ls
             print*,'--->  A: ',AG,BG,CG,DG
             print*,'---> dX: ',u1o,v1o,u2o,v2o
             print*,'--->   : ',dbar,dmax
             stop '2MASS FAILURE!'
             endif
         write(*,'('' '')')
         write(*,'(''---> MSORTs: '',i4)') MSORTs
         write(*,'(''--->     Us: '',i4)') Us
         write(*,'(''--->     Lu: '',i4)') Lu
         write(*,'(''--->     Ls: '',i4)') Ls
         write(*,'(''--->  NMATs: '',i4)') NMATs
         write(*,'(''--->     AB: '',2f10.6)') AG, BG
         write(*,'(''--->     CD: '',2f10.6)') CG, DG
         write(*,'(''--->    SCL: '',1f10.6)') sqrt(AG*DG-BG*CG)
         write(*,'(''--->    ANG: '',1f10.6)') 180/3.14*atan2(BG-CG,
     .                                                        AG+DG)
         write(*,'(''--->    uv1: '',2f10.2)') u1o,v1o
         write(*,'(''--->    uv2: '',2f10.2)') u2o,v2o
         write(*,'(''--->    uvD: '',2f10.2)') u1o-u2o,v1o-v2o
         write(*,'(''--->   dbar: '',2f10.2)') dbar
         write(*,'(''--->   dmax: '',2f10.2)') dmax
         write(*,'('' '')')
         do N = 1, Ns
c           uu_n(N) = u2o + AG*(u_n(N)-u1o) + BG*(v_n(N)-v1o)
c           vv_n(N) = v2o + CG*(u_n(N)-u1o) + DG*(v_n(N)-v1o)
            uu_n(N) = u_n(N) + (u2o-u1o)
            vv_n(N) = v_n(N) + (v2o-v1o)
            enddo
         write(HH(HHS+001),'(''#--------------------------------'')')
         write(HH(HHs+002),197) Us,NMATs,AG,BG,CG,DG,
     .                                   sqrt(AG*DG-BG*CG),
     .                                   180/3.14*atan2(BG-CG,AG+DG),
     .                                   u1o-u2o,v1o-v2o,dbar,dmax
  197    format('# 2MASS MATCH SUMMARY --- ',i4,1x,i4,1x,4f10.6,
     .                                       5x,f10.6,1x,f10.6,
     .                                       5x,2f7.2,5x,2f7.2)
         write(HH(HHS+003),'(''#--------------------------------'')')
         write(HH(HHs+004),'(''#    NOVLP: '',i4  )') Us
         write(HH(HHs+005),'(''#    NMATs: '',i4  )') NMATs
         write(HH(HHs+006),'(''#       AB: '',2f10.6)') AG,BG
         write(HH(HHs+007),'(''#       CD: '',2f10.6)') DG,DG
         write(HH(HHs+008),'(''#      SCL: '',1f10.6)')
     .                                        sqrt(AG*DG-BG*CG)
         write(HH(HHs+009),'(''#      ANG: '',1f10.6)')
     .                                        180/3.14*atan2(BG-CG,
     .                                                       AG+DG)
         write(HH(HHs+010),'(''#      uv1: '',2f10.2)') u1o,v1o
         write(HH(HHs+011),'(''#      uv2: '',2f10.2)') u2o,v2o
         write(HH(HHs+012),'(''#      uvD: '',2f10.2)') u1o-u2o,
     .                                                  v1o-v2o
         write(HH(HHs+013),'(''#     dbar: '',2f10.2)') dbar
         write(HH(HHs+014),'(''#     dmax: '',2f10.2)') dmax
         write(HH(HHs+015),'(''# '')')
         write(HH(HHs+016),198)
         HHs = HHs + 16
         do NMAT = 1, NMATs
            N = n_msort(n1mat(NMAT))
            O = oo_u(n2mat(NMAT))
            n_NMAT(NMAT) = N
            o_NMAT(NMAT) = O
            HHs = HHs + 1
            write(HH(HHs),199) NMAT,ra_o(O),de_o(O),
     .                              mj_o(O),mh_o(O),mk_o(O),aa_o(O),
     .                              ug_o(O),vg_o(O),
     .                               x_n(N), y_n(N), m_n(N),
     .                              xx_n(N),yy_n(N),
     .                               u_n(N), v_n(N),mm_n(N),
     .                              uu_n(N),vv_n(N),N,
     .                              uu_n(N)-ug_o(O),vv_n(N)-vg_o(O)
            write( *,'(''HH'',i3.3,1x,249a)') HHs,HH(HHs)
  199       format('# CATMAT',i4.4,
     .                        2x,f10.6,1x,f10.6,1x,
     .                        f5.2,1x,f5.2,1x,f5.2,1x,a3,1x,
     .                        f9.2,1x,f9.2,
     .                        4x,f9.2,1x,f9.2,1x,f8.3,1x,
     .                           f9.2,1x,f9.2,1x,
     .                           f9.2,1x,f9.2,1x,f8.3,3x,
     .                           f9.2,1x,f9.2,1x,i6.6,
     .                        4x,f7.2,1x,f7.2)
 
  198       format('#       NMAT    RA_CAT     DEC_CAT   ',
     .             ' MJ    MH    MK  QQQ     U_CAT     V_CAT       ',
     .             '  xraw      yraw     mraw      xcor      ycor  ',
     .             '    u         v       M         ',
     .             '    U         V        N        du      dv ')
            enddo
         HHs = HHs + 1
         write(HH(HHs),198)
         HHs = HHs + 1
         write(HH(HHs),'(''# '')')
         if (CATREG(1:4).eq.'AUTO') then
            FILEu = PREFIX(1:9) // '_cat.reg'
            open(98,file=FILEu,status='unknown')
            write(98,211)
            write(98,212)
            write(98,'(''physical'')')
            do NMAT = 1, NMATs
               write(98,982) ug_o(o_NMAT(NMAT)),vg_o(o_NMAT(NMAT))
               write(98,983) uu_n(n_NMAT(NMAT)),vv_n(n_NMAT(NMAT))
  982          format('circle(',f9.2,',',f9.2,
     .                ',5.0) # width=3 color=green')
  983          format('circle(',f9.2,',',f9.2,
     .                ',1.0) # width=3 color=green')
               enddo
            do O = 1, Os
               write(98,981) ug_o(O), vg_o(O)
  981          format('circle(',f9.2,',',f9.2,
     .                ',5.0) # width=1 color=yellow')
               enddo
            do MSORT = 1, MSORTs
               write(98,984) u_msort(MSORT) + (u2o-u1o),
     .                       v_msort(MSORT) + (v2o-v1o)
  984          format('circle(',f9.2,',',f9.2,
     .                ',1.0) # width=1 color=yellow')
               enddo
            close(98)
            endif
         endif
 
 
      GA =  GD/(AG*DG-BG*CG)
      GB = -GB/(AG*DG-BG*CG)
      GC = -GC/(AG*DG-BG*CG)
      GD =  GA/(AG*DG-BG*CG)
 
 
 
c---------------------------------------------------------------
c
c output the final boundary info...
c
      write(HH(HHS+1),'(''#-------------------------------------'')')
      write(HH(HHs+2),'(''# BOUNDARY INFO   CHIP='',i1,'' Ks='',i1)')
     .                      INSTu,Ks
      write(HH(HHS+3),'(''#-------------------------------------'')')
      write(HH(HHs+4),442)
      write(HH(HHs+5),'(''# '')')
      HHs = HHs + 6
      do K = 1, Ks
      do P = 1, 4
         BDRY_UU(K,P) = BDRY_UG(K,P) + (u2o-u1o)
         BDRY_VV(K,P) = BDRY_VG(K,P) + (v2o-v1o)
         write(HH(HHs),443) K,P,BDRY_XR(P,K),BDRY_YR(P,K),
     .                          BDRY_XC(P,K),BDRY_YC(P,K),
     .                          BDRY_UG(P,K),BDRY_VG(P,K),
     .                          BDRY_RA(P,K),BDRY_DE(P,K),
     .                          BDRY_UU(P,K),BDRY_VV(P,K)
         HHs = HHs + 1
         enddo
         enddo
 
      if (SHOW_USE(1:6).eq.'GCTEMP') then
          if (VERBOSE.ge.2) then
              print*,' '
              print*,'SHOW_USE: ',SHOW_USE
              print*,' '
              endif
          UMIN =  99999
          UMAX = -99999
          VMIN =  99999
          VMAX = -99999
          do K = 1, Ks
          do P = 1, 4
             UMIN = min(UMIN,int(BDRY_XC(P,K)))
             UMAX = max(UMAX,int(BDRY_XC(P,K)))
             VMIN = min(VMIN,int(BDRY_YC(P,K)))
             VMAX = max(VMAX,int(BDRY_YC(P,K)))
             enddo
             enddo
          UMIN = UMIN-5
          UMAX = UMAX+5
          VMIN = VMIN-5
          VMAX = VMAX+5
          if (VERBOSE.ge.2) then
             print*,'   UMIN/MAX: ',UMIN,UMAX
             print*,'   VMIN/MAX: ',VMIN,VMAX
             endif
          NAXISU = 1 + UMAX-UMIN
          NAXISV = 1 + VMAX-VMIN
          if (VERBOSE.ge.2) then
             print*,'     NAXISU: ',NAXISU
             print*,'     NAXISV: ',NAXISV
             endif
          allocate(pixr(NAXISU,NAXISV))
          do i = 0001, NAXISU
          do j = 0001, NAXISV
             pixr(i,j) = 0.
             ug = UMIN + (i-1)
             vg = VMIN + (j-1)
             Ku = 0
             do k = 1, Ks
                 if (inside_poly(ug,vg,BDRY_XC(1,K),BDRY_YC(1,K),4))
     .               Ku = K
                 enddo
             if (Ku.ne.0) then
c               dRA = CD1_1_OUTP*(ug-CRPIX1_OUTP)
c   .               + CD1_2_OUTP*(vg-CRPIX2_OUTP)
c               dDE = CD2_1_OUTP*(ug-CRPIX1_OUTP)
c    .              + CD2_2_OUTP*(vg-CRPIX2_OUTP)
c               RA  = xy2r(dRA,dDE,CRVAL1_OUTP,CRVAL2_OUTP)
c               DE  = xy2d(dRA,dDE,CRVAL1_OUTP,CRVAL2_OUTP)
c               dRA = rd2x(RA,DE,CRVAL1_INST,CRVAL2_INST)
c               dDE = rd2y(RA,DE,CRVAL1_INST,CRVAL2_INST)
c               xgc = xgc0 + RCD1_1_INST*dRA + RCD1_2_INST*dDE
c               ygc = ygc0 + RCD2_1_INST*dRA + RCD2_2_INST*dDE
                xgc = ug
                ygc = vg
                call xcyc2xryr_stdgc(xgc,ygc,xr,yr,GDCFILE_USE)
                ii = int(xr+0.5)
                jj = int(yr+0.5)
                if (ii.ge.0001.and.ii.le.NAXIS1.and.
     .              jj.ge.0001.and.jj.le.NAXIS2) then
                    pixr(i,j) = pixc(ii,jj)*350.00/EXPTu
                    endif
                if (j.eq.NAXISV/2.and.i.eq.i/250*250)
     .              write(*,913) i,j,ug,vg,pixr(i,j),
     .                               xgc,ygc,xr,yr,dRA,dDE
  913           format(2i5,1x,2f10.1,5x,f8.1,
     .                     5x,2f10.2,5x,2f10.2,5x,2f10.5)
                endif
             enddo
             enddo
 
          SHOW_USEu = PREFIX(1:9) // '_geo.fits'
          call writfits_r4_WCSLTV(SHOW_USEu,pixr,NAXISU,NAXISV,
     .                            xgc0-(UMIN-1),
     .                            ygc0-(VMIN-1),
     .                            CRVAL1_OUTP, CRVAL2_OUTP,
     .                            CD1_1_INST,  CD1_2_INST,
     .                            CD2_1_INST,  CD2_2_INST,
     .                            -UMIN+1.0d0-(u2o-u1o),
     .                            -VMIN+1.0d0-(v2o-v1o),
     .                            1.0d0,0.0d0,0.0d0,1.0d0)
 
          print*,'   xgc0: ',xgc0,ygc0
          print*,'   UMIN: ',UMIN,VMIN
          print*,' CRPIX1: ',xgc0-(UMIN-1), ygc0-(VMIN-1)
          print*,' CRPIX2: ',xgc0+(UMIN-1), ygc0+(VMIN-1)
          deallocate(pixr)
          endif
 
 
      if (SHOW_REF(1:4).eq.'AUTO') then
         IMREFu = PREFIX(1:9) // '_ref.fits'
         if (As.ne.0) IMREFu = PREFIX(1:9) // '_rfa.fits'
         print*,'OUTPUT pix_ref: ',IMREFu
         UMIN =  99999
         UMAX = -99999
         VMIN =  99999
         VMAX = -99999
         do K = 1, Ks
         do P = 1, 4
            UMIN = min(UMIN,int(BDRY_UU(P,K)))
            UMAX = max(UMAX,int(BDRY_UU(P,K)))
            VMIN = min(VMIN,int(BDRY_VV(P,K)))
            VMAX = max(VMAX,int(BDRY_VV(P,K)))
            enddo
            enddo
         if (VERBOSE.ge.2) then
            print*,'   UMIN/MAX: ',UMIN,UMAX
            print*,'   VMIN/MAX: ',VMIN,VMAX
            endif
         NAXISU = 1 + UMAX-UMIN
         NAXISV = 1 + VMAX-VMIN
         allocate(pixr(NAXISU,NAXISV))
 
         if (VERBOSE.ge.2) then
            write(*,'(''RCD_INST: '',4f20.10)') RCD1_1_INST,RCD1_2_INST,
     .                                          RCD2_1_INST,RCD2_2_INST
            write(*,'('' CD_INST: '',4f20.10)')  CD1_1_INST, CD1_2_INST,
     .                                           CD2_1_INST, CD2_2_INST
            endif
         write(66,*) '  REF CRPIX1_INST: ',CRPIX1_INST
         write(66,*) '  REF CRPIX2_INST: ',CRPIX2_INST
         write(66,*) '  REF CRVAL1_INST: ',CRVAL1_INST
         write(66,*) '  REF CRVAL2_INST: ',CRVAL2_INST
         print*,'    Ks: ',Ks
         print*,'NAXISU: ',NAXISU
         print*,'NAXISV: ',NAXISV
         do i = 0001, NAXISU
         do j = 0001, NAXISV
            pixr(i,j) = 0.
            ug = UMIN + (i-1)
            vg = VMIN + (j-1)
            Ku = 0
            do k = 1, Ks
                if (inside_poly(ug,vg,BDRY_UU(1,K),BDRY_VV(1,K),4))
     .              Ku = K
                enddo
            if (KSEL.ne.0.and.Ku.ne.KSEL) Ku = 0
            if (Ku.ne.0) then
               dRA = CD1_1_OUTP*(ug-CRPIX1_OUTP)
     .             + CD1_2_OUTP*(vg-CRPIX2_OUTP)
               dDE = CD2_1_OUTP*(ug-CRPIX1_OUTP)
     .             + CD2_2_OUTP*(vg-CRPIX2_OUTP)
               RA  = xy2r(dRA,dDE,CRVAL1_OUTP,CRVAL2_OUTP)
               DE  = xy2d(dRA,dDE,CRVAL1_OUTP,CRVAL2_OUTP)
               dRA = rd2x(RA,DE,CRVAL1_INST,CRVAL2_INST)
               dDE = rd2y(RA,DE,CRVAL1_INST,CRVAL2_INST)
               xgc = xgc0 + RCD1_1_INST*dRA + RCD1_2_INST*dDE
               ygc = ygc0 + RCD2_1_INST*dRA + RCD2_2_INST*dDE
               call xcyc2xryr_stdgc(xgc,ygc,xr,yr,GDCFILE_USE)
               ii = int(xr+0.5)
               jj = int(yr+0.5)
               if (ii.ge.0001.and.ii.le.NAXIS1.and.
     .             jj.ge.0001.and.jj.le.NAXIS2) then
                   pixr(i,j) = pixc(ii,jj)
                   endif
               if (j.eq.NAXISV/2.and.i.eq.i/250*250)
     .             write(*,113) i,j,ug,vg,pixr(i,j),
     .                              xgc,ygc,xr,yr,dRA,dDE
  113          format(2i5,1x,2f10.1,5x,f8.1,
     .                    5x,2f10.2,5x,2f10.2,5x,2f10.5)
               endif
            enddo
            enddo
         call writfits_r4_WCSLTV(IMREFu,pixr,NAXISU,NAXISV,
     .                           CRPIX1_OUTP-UMIN+1-(u2o-u1o),
     .                           CRPIX2_OUTP-VMIN+1-(v2o-v1o),
     .                           CRVAL1_OUTP,CRVAL2_OUTP,
     .                           CD1_1_OUTP, CD1_2_OUTP,
     .                           CD2_1_OUTP, CD2_2_OUTP,
     .                           -UMIN+1.0d0-(u2o-u1o),
     .                           -VMIN+1.0d0-(v2o-v1o),
     .                           1.0d0,0.0d0,0.0d0,1.0d0)
         if (SHOW_ALL(1:3).eq.'YES')
     .   call writfits_r4('pixr.fits',pixr,NAXISU,NAXISV)
         deallocate(pixr)
         endif
 
 
      if (DOSTARDB) then
          FILEu = PREFIX(1:13) // '.stardb_xym'
          open(58,file=FILEu,status='unknown')
          do N = 1, Ns
            xr = x_n(N)
            yr = y_n(N)
            sr = s_n(N)
            mr = m_n(N)
            fr = 10**(-m_n(N)/2.5)
            qr = q_n(N)
            ii = int(xr+0.5)
            jj = int(yr+0.5)
            xru = xr         ! need real*4
            yru = yr
            call locpsfij_stdpsf(ii,jj,psfloc,PSFFILE_USE)
            atot = (pixc(ii-2,jj-2)+pixc(ii-2,jj-1)
     .             +pixc(ii-1,jj-2)+pixc(ii-1,jj-1)
     .             +pixc(ii+1,jj+2)+pixc(ii+1,jj+1)
     .             +pixc(ii+2,jj+2)+pixc(ii+2,jj+1)-8*sr)
            btot = (pixc(ii-2,jj+2)+pixc(ii-2,jj+1)
     .             +pixc(ii-1,jj+2)+pixc(ii-1,jj+1)
     .             +pixc(ii+1,jj-2)+pixc(ii+1,jj-1)
     .             +pixc(ii+2,jj-2)+pixc(ii+2,jj-1)-8*sr)
            aexp = (rpsf_phot(ii-2-xru,jj-2-yru,psfloc)
     .             +rpsf_phot(ii-2-xru,jj-1-yru,psfloc)
     .             +rpsf_phot(ii-1-xru,jj-2-yru,psfloc)
     .             +rpsf_phot(ii-1-xru,jj-1-yru,psfloc)
     .             +rpsf_phot(ii+1-xru,jj+2-yru,psfloc)
     .             +rpsf_phot(ii+1-xru,jj+1-yru,psfloc)
     .             +rpsf_phot(ii+2-xru,jj+2-yru,psfloc)
     .             +rpsf_phot(ii+2-xru,jj+1-yru,psfloc))
            bexp = (rpsf_phot(ii-2-xru,jj+2-yru,psfloc)
     .             +rpsf_phot(ii-2-xru,jj+1-yru,psfloc)
     .             +rpsf_phot(ii-1-xru,jj+2-yru,psfloc)
     .             +rpsf_phot(ii-1-xru,jj+1-yru,psfloc)
     .             +rpsf_phot(ii+1-xru,jj-2-yru,psfloc)
     .             +rpsf_phot(ii+1-xru,jj-1-yru,psfloc)
     .             +rpsf_phot(ii+2-xru,jj-2-yru,psfloc)
     .             +rpsf_phot(ii+2-xru,jj-1-yru,psfloc))
            cexp =  rpsf_phot(ii  -xru,jj  -yru,psfloc)
            pabs = 0.
            ptot = 0.
            ftot = 0.
            do iii = -5, 5
            do jjj = -5, 5
               if (iii**2+jjj**2.ge.3**2.and.
     .             iii**2+jjj**2.le.5.5**2) then
                   f = rpsf_phot(ii+iii-xru,ii+iii-yru,psfloc)
                   ftot = ftot + f
                   ptot = ptot + (pixc(ii+iii,jj+jjj)-sr)
                   pabs = pabs + abs(pixc(ii+iii,jj+jjj)-sr-fr*f)
                   endif
               enddo
               enddo
            g1 = (ptot-fr*ftot)/fr
            g2 = pabs/fr
            if (g1.gt. 9.999) g1 =  9.999
            if (g1.lt.-9.999) g1 = -9.999
            if (g2.gt. 9.999) g2 =  9.999
            if (g2.lt.-9.999) g2 = -9.999
            write(58,258)
     .            dclip(xr,0000.0d0,9999.0d0),
     .            dclip(yr,0000.0d0,9999.0d0),
     .            dclip(mr, -99.9d0,   0.0d0),
     .            dclip(qr*1.0d0,-0.9d0,   9.9d0),
     .            fr,   sr,
     .            pixc(ii,jj)+0.5, cexp,
c    .            atot, aexp,
c    .            btot, bexp,
     .            N, sat_n(N), ' ',
     .            g1, g2
 258        format(f8.3,1x,f8.3,1x,f8.4,1x,f8.5,
     .             3x,f10.2,1x,f8.2,
     .             3x,f10.1,1x,f8.5,
c    .             3x,f10.1,1x,f8.5,
c    .             3x,f10.1,1x,f8.5,
     .             1x,'N',i5.5,1x,i1,1a,
     .             3x,f8.5,1x,f8.5)
            enddo!N
          close(58)
 
          FILEu = PREFIX(1:13) // '.stardb_ras'
          open(58,file=FILEu,status='unknown')
          do N = 1, Ns
             ii = i_n(N)
             jj = j_n(N)
             call locpsfij_stdpsf(ii,jj,psfloc,PSFFILE_USE)
             do i = max(1,ii-5),min(NAXIS1,ii+5)
             do j = max(1,jj-5),min(NAXIS2,jj+5)
                dx = i-xr
                dy = j-yr
                xr = x_n(N)
                yr = y_n(N)
                fr = 10**(-m_n(N)/2.5)
                sr = s_n(N)
                write(58,158) i,j,pixc(i,j),
     .                        xr,yr,fr,sr,
     .                        rpsf_phot(dx,dy,psfloc), N
 158            format(i4,1x,i4,1x,f13.2,1x,
     .                 f8.2,1x,f8.2,1x,f12.0,1x,f9.3,f10.6,
     .                 1x,'N',i5.5)
                enddo
                enddo
             enddo
          close(58)
 
          endif
 
 
 
      deallocate(pixo)
      deallocate(pixp)
      deallocate(pixx)
      deallocate(pixc)
      deallocate(pixq)
      deallocate(pixn)
      if (SHOW_MSK(1:1).ne.'X') deallocate(pixm)
 
      if (INSTu.eq.0) deallocate(pixw)
 
      if (SHOW_ALL(1:3).eq.'YES') stop 'SHOW_ALL only shows one'
 
      CRPIX1_DEST = CRPIX2_OUTP
      CRPIX2_DEST = CRPIX2_OUTP
      CRVAL1_DEST = CRVAL1_OUTP
      CRVAL2_DEST = CRVAL2_OUTP
      CD1_1_DEST  = CD1_1_OUTP
      CD1_2_DEST  = CD1_2_OUTP
      CD2_1_DEST  = CD2_1_OUTP
      CD2_2_DEST  = CD2_2_OUTP
 
      if (VERBOSE.ge.1) then
          print*,' '
          print*,'SEPARATE READ-IN: '
          print*,'       CRVAL1: ',CRVAL1_INST,' ' //
     .                                         ra2sexig(CRVAL1_INST)
          print*,'       CRVAL2: ',CRVAL2_INST,de2sexig(CRVAL2_INST)
          print*,' '
          endif
 
      FILTu = FILT_INST
      EXPTu = EXPT_INST
      RDATu = RDAT_INST
      PROPu = PROP_INST
 
      return
 
 
  211 format('# Region file format: DS9 version 4.1')
  212 format('global color=green dashlist=8 3 width=1',
     .       ' font="helvetica 10 normal" select=1 highlite=1',
     .       ' dash=0 fixed=0 edit=1 move=1 delete=1 include=1',
     .       ' source=1')
  214 format('circle(',f11.7,',',f11.7,',0.05")')                     ! UNSAT rd
  215 format('circle(',f11.7,',',f11.7,',0.05") # width=4 color=red') !   SAT rd
  216 format('circle(',f9.2,',',f9.2,',',f5.2,')')                     ! UNSAT xy,XY,uv
  217 format('circle(',f9.2,',',f9.2,',',f5.2,') # width=4 color=red') !   SAT xy,XY,uv
 
      end
 
 
 
 
 
 
 
 
c-------------------------------------------------
c
c
c
      subroutine fitsat_ir(xx,yy,ss,zz,qq,
     .                     psfloc,pixc,NAXIS1,NAXIS2,
     .                     LOFLAG,HIFLAG)
      implicit none
 
      real*8  xx, yy
      real*8  ss, zz
      real    qq
      integer LOFLAG
      integer HIFLAG
      integer NAXIS1
      integer NAXIS2
      real    psfloc(101,101)
      real    pixc(NAXIS1,NAXIS2)
 
      integer rsat
      integer Ns
 
      integer U, Us
      real    pu(9999)
      integer iu(9999)
      integer ju(9999)
      real    fu(9999),  fuu(9999)
      real   dxu(9999), dxuu(9999)
      real   dyu(9999), dyuu(9999)
 
      integer ii,jj
      integer i, j
      real    dx, dy , rpsf_phot
      real*8  ptot, ftot, etot
 
      integer NCOUNT
      data    NCOUNT/0/
 
      real xmin
      real ymin
      real zmin
      real emin
 
      integer idx, idy
      real xc, yc
 
      ii = int(xx+0.5)
      jj = int(yy+0.5)
      qq = 0.999
      zz = pixc(ii,jj)
 
      rsat = 3
    1 continue
      Ns = 0
      Us = 0
      do i = -rsat, rsat
      do j = -rsat, rsat
         if (i**2+j**2.le.(rsat+0.5)**2) then
             if (ii+i.lt.000001+1) return
             if (jj+j.lt.000001+1) return
             if (ii+i.gt.NAXIS1-1) return
             if (jj+j.gt.NAXIS2-1) return
             Ns = Ns + 1
             if (pixc(ii+i  ,jj+j  ).lt.HIFLAG.and.
     .           pixc(ii+i-1,jj+j  ).lt.HIFLAG.and.
     .           pixc(ii+i+1,jj+j  ).lt.HIFLAG.and.
     .           pixc(ii+i  ,jj+j-1).lt.HIFLAG.and.
     .           pixc(ii+i  ,jj+j+1).lt.HIFLAG.and.
     .           pixc(ii+i-1,jj+j+1).lt.HIFLAG.and.
     .           pixc(ii+i+1,jj+j+1).lt.HIFLAG.and.
     .           pixc(ii+i-1,jj+j-1).lt.HIFLAG.and.
     .           pixc(ii+i+1,jj+j-1).lt.HIFLAG) then
                 Us = Us + 1
                 pu(Us) = pixc(ii+i,jj+j)-ss
                 iu(Us) = ii+i
                 ju(Us) = jj+j
                 endif
             endif
         enddo
         enddo
 
      if (Us.lt.Ns*0.50) then
          rsat = rsat + 1
          if (rsat.lt.10) goto 1
          endif
 
      if (rsat.ge.10.and.Us.lt.Ns*0.075) return
 
      xmin = ii
      ymin = jj
      emin = 9e9
      zmin = 0.0
      do idx = -20, 20
      do idy = -20, 20
         xc = ii + idx*0.05
         yc = jj + idy*0.05
         do U = 1, Us
            dxu(U) = iu(U)-xc
            dyu(U) = ju(U)-yc
            fu(U) = rpsf_phot(dxu(U),dyu(U),psfloc)
            enddo
         ptot = 0.
         ftot = 0.
         do U = 1, Us
            ptot = ptot + pu(U)-ss
            ftot = ftot + fu(U)
            enddo
         zz = ptot/ftot
         etot = 0
         do U = 1, Us
            etot = etot + abs(pu(U)-zz*fu(U))
            enddo
         if (etot.lt.emin) then
            emin = etot
            xmin = xc
            ymin = yc
            zmin = zz
            do U = 1, Us
                fuu(U) =  fu(U)
               dxuu(U) = dxu(U)
               dyuu(U) = dyu(U)
               enddo
            endif
         enddo
         enddo
 
         if (ftot.le.0) then
             zz = 0.
             qq = 0.
             return
             endif
 
 
      if (.false.) then
         NCOUNT = NCOUNT+1
         do U = 1, Us
            write(81,181) NCOUNT, U, iu(U), ju(U),
     .                    dxuu(U), dyuu(U), fuu(U),
     .                    pu(U), zz
  181       format(1x,i7.7,1x,i5.5,1x,i4,1x,i4,
     .                         3x,f6.2,1x,f6.2,1x,f8.6,
     .                         3x,f8.1,1x,f15.1)
            enddo
         endif
 
      xx = xmin
      yy = ymin
      zz = zmin
      qq = emin/ptot
 
      return
      end
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/HST1PASS/ROUTINES/fill_charout.f"
c****                                         
c*********************************************
                                              
 
 
 
c-------------------------------------------------------------
c
c fill out the array of possible output values (into a string)
c
c
      subroutine fill_charout(CHAROUT_F,
     .                         u, v,mm,
     .                         x, y, m, k,
     .                        xx,yy, h,hh,
     .                         r, d,
     .                         i, j, p, q, c, s, uu, vv, n)
      implicit none
 
      character*20 CHAROUT_F(_FIELDs_)
      real*8   u,  v, mm
      real*8   x,  y,  m
      integer  k
      real*8  xx, yy
      integer  h
      real    hh
      real*8   r,  d
      integer  i,  j
      real     p
      real     q,  c, s
      real*8  uu, vv
      integer n
 
      character*11 ra2sexig
      character*12 de2sexig
 
      integer F, o
 
      write(CHAROUT_F(01),'(f14.4)') u
      write(CHAROUT_F(02),'(f14.4)') v
      write(CHAROUT_F(03),'(f08.3)') mm
      write(CHAROUT_F(04),'(f11.4)') x
      write(CHAROUT_F(05),'(f11.4)') y
      write(CHAROUT_F(06),'(f08.3)') m
      write(CHAROUT_F(07),'(i02.1)') k
      write(CHAROUT_F(08),'(f11.4)') xx
      write(CHAROUT_F(09),'(f11.4)') yy
      write(CHAROUT_F(10),'(i02.1)') h
      write(CHAROUT_F(11),'(f08.2)') min(hh,99.)
      write(CHAROUT_F(12),'(f15.7)') r
      write(CHAROUT_F(13),'(f15.7)') d
      write(CHAROUT_F(14),'(11a  )') ra2sexig(r)
      write(CHAROUT_F(15),'(12a  )') de2sexig(d)
      write(CHAROUT_F(16),'(i6   )') i
      write(CHAROUT_F(17),'(i6   )') j
      write(CHAROUT_F(18),'(f12.2)') p
      write(CHAROUT_F(19),'(f12.3)') q
      write(CHAROUT_F(20),'(f12.3)') c
      write(CHAROUT_F(21),'(f14.4)') uu
      write(CHAROUT_F(22),'(f14.4)') vv
      write(CHAROUT_F(23),'(i1.1)') k
      write(CHAROUT_F(24),'(''K'',i1.1)') k
      write(CHAROUT_F(25),'(f10.2)') s
      write(CHAROUT_F(26),'(i6.6)') n
 
      do F = 1, _FIELDs_
         do o = 01, 20
            if (CHAROUT_F(F)(o:o).eq.'*') CHAROUT_F(F)(o:o) = '0'
            enddo
         enddo
 
      return
      end
 
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/HST1PASS/ROUTINES/find_rdate.f"
c****                                         
c*********************************************
                                              
 
      subroutine find_rdate(FILENAME,rDAT,TIMESTR,DATESTR)
      implicit none
 
      character*80 FILENAME
      real         rDAT
      character*20 TIMESTR
      character*20 DATESTR
 
      character*20 DATESTRIN
 
      integer iRAH, iRAM, iRAS, iFRA
      integer iDEG, iMIN, iSEC, iFDE
      integer iDAY, iMON, iYIR, iHRS
 
      integer DOY(12)
      data DOY / 000, 031, 059, 090, 120, 151,
     .           181, 212, 243, 273, 304, 334/
 
      character*20 STREAM
 
      iMON = 1
      iDAY = 1
      iYIR = 0
 
      iHRS = 0
      iMIN = 0
      iSEC = 0
 
      TIMESTR = 'TIMELESS'
      DATESTR = 'DATELESS'
 
      call query_hdr(FILENAME,'TIME-OBS ',stream)
      TIMESTR   = stream(02:09)
      call query_hdr(FILENAME,'DATE-OBS ',stream)
      DATESTRIN = stream(02:20)
 
c     print*,'DATESTRIN: ',DATESTRIN
      if (DATESTRIN(1:1).eq.'''') DATESTRIN = DATESTRIN(2:20)
c     print*,'DATESTRIN: ',DATESTRIN
      if (DATESTRIN(3:3).eq.'/') DATESTR(1:8) = DATESTRIN(1:8)
      if (DATESTRIN(5:5).eq.'-') then
          DATESTR(1:2) = DATESTRIN(9:10)
          DATESTR(3:3) = '/'
          DATESTR(4:5) = DATESTRIN(6:7)
          DATESTR(6:6) = '/'
          DATESTR(7:8) = DATESTRIN(3:4)
          endif
 
c     print*,'DATESTR  : ',DATESTR
      if (DATESTR.ne.'DATELESS') then
          read(DATESTR(1:2),*) iDAY
          read(DATESTR(4:5),*) iMON
          read(DATESTR(7:8),*) iYIR
          endif
 
c     print*,'TIMESTR  : ',TIMESTR
      if (TIMESTR.ne.'TIMELESS'.and.
     .    TIMESTR(3:3).eq.':'.and.
     .    TIMESTR(6:6).eq.':') then
          read(TIMESTR(1:2),*) iHRS
          read(TIMESTR(4:5),*) iMIN
          read(TIMESTR(7:8),*) iSEC
          endif
 
      rDAT = iYIR + (DOY(iMON)+iDAY)/365.0
      rDAT = rDAT + (iHRS + iMIN/60. + iSEC/60./60.)/8766.
 
      if (rDAT.gt.050) rDAT = rDAT - 100.0
 
c     print*,' '
c     print*,'FIND rDAT...'
c     print*,'  1) rDAT: ',rDAT
c     print*,'  2) iDMY: ',iDAY,iMON,iYIR
c     print*,'  3) iHMS: ',iHRS,iMIN,iSEC
c     print*,' '
 
      return
      end
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/HST1PASS/ROUTINES/lnc.f"
c****                                         
c*********************************************
                                              
 
      integer function LNC(STRING,CMAX)
      implicit none
 
      character(*) STRING
      integer CMAX
 
      LNC = CMAX
    1 continue
      if (STRING(LNC:LNC).ne.' ') return
      if (LNC.gt.1) then
         LNC = LNC - 1
         goto 1
         endif
      return
 
      end
 
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/HST1PASS/ROUTINES/output_lists.f"
c****                                         
c*********************************************
                                              
 
 
      subroutine output_lists(NLISTs,NITEMSO_NL,OUTLIST_NL,
     .                        OUTDIR,PREFIX,
     .                        HH, HHs,
     .                        Ns,  u_n,  v_n, mm_n,
     .                             x_n,  y_n,  m_n,  k_n,
     .                            xx_n, yy_n,
     .                            uu_n, vv_n,
     .                             h_n, hh_n,
     .                             r_n,  d_n,
     .                             i_n,  j_n,  p_n,
     .                             q_n,  c_n,  s_n)
      implicit none
 
 
 
      integer       NLISTs
      character*80  OUTLIST_NL(9)  ! the suffix of the output file
      integer       NITEMSO_NL(9)  ! the number of elements to output
      character*80  OUTDIR
      character*80  PREFIX
 
      integer       HHs
      character*249 HH(999)   ! max 999 header pages!
 
      integer        Ns
      real*8         u_n(_NSTMAX_),  v_n(_NSTMAX_), mm_n(_NSTMAX_)
      real*8         x_n(_NSTMAX_),  y_n(_NSTMAX_),  m_n(_NSTMAX_)
      real*8        xx_n(_NSTMAX_), yy_n(_NSTMAX_)
      real*8        uu_n(_NSTMAX_), vv_n(_NSTMAX_)
      integer        k_n(_NSTMAX_)
      integer        h_n(_NSTMAX_)
      real          hh_n(_NSTMAX_)
      real*8         r_n(_NSTMAX_),  d_n(_NSTMAX_)
      integer        i_n(_NSTMAX_),  j_n(_NSTMAX_)
      real           p_n(_NSTMAX_)
      real           q_n(_NSTMAX_),  c_n(_NSTMAX_)
      real           s_n(_NSTMAX_)
c
c----------------------------------------------------------
c
      integer         N
      character*300   STRING300, STRING300H
      integer         F, F_NITEM(99)
      character*4     CNITEM
      integer         CLAST
      integer         C1_F(_FIELDs_), N1_NITEM(99), N1
      integer         C2_F(_FIELDs_), N2_NITEM(99), N2
      character*23 CHAR_HEADER(_FIELDs_)
      data  CHAR_HEADER / 'u -- u ref-frame       ', ! 01
     .                    'v -- v ref-frame       ', ! 02
     .                    'M -- instl mag (1000s) ', ! 03
     .                    'x -- raw detector x pos', ! 04
     .                    'y -- raw detector y pos', ! 05
     .                    'm -- instl mag         ', ! 06
     .                    'k -- detector chip no  ', ! 07
     .                    'X -- dist-cor x posn   ', ! 08
     .                    'Y -- dist-cor y posn   ', ! 09
     .                    'h -- local isolation   ', ! 10
     .                    'H -- list-based isolatn', ! 11
     .                    'r -- RA (deg)          ', ! 12
     .                    'd -- Dec (deg)         ', ! 13
     .                    'R -- RA (sexig)        ', ! 14
     .                    'D -- Dec (sexig)       ', ! 15
     .                    'i -- peak pixel column ', ! 16
     .                    'j -- peak pixel row    ', ! 17
     .                    'p -- peak pixel value  ', ! 18
     .                    'q -- quality of fit    ', ! 19
     .                    'c -- chisq of fit      ', ! 20
     .                    'U -- u ref-frame (corr)', ! 21
     .                    'V -- v ref-frame (corr)', ! 22
     .                    'k -- chip number       ', ! 23
     .                    'K -- chip number (w/K) ', ! 24
     .                    's -- sky value         ', ! 25
     .                    'n -- star number       '/ ! 26
 
      integer      NITEM
      integer      NLIST
      character*20 CHAROUT_F(_FIELDs_)
      character*80 FILEOUT
      integer      i
      integer      LNC
 
c--------------------------
c
c output the star lists
c
      print*,'OUTPUT LISTS...'
      print*,'NLISTs: ',NLISTs
 
      if (NLISTs.eq.0) return
 
      print*,'SETUPa--- ',_FIELDs_
      do F = 1, _FIELDs_
         C1_F(F) = 20
         C2_F(F) = 01
         enddo
      print*,'SETUPb --- ',Ns
      do N = 1, Ns
         call fill_charout(CHAROUT_F,
     .                      u_n(N), v_n(N),mm_n(N),
     .                      x_n(N), y_n(N), m_n(N), k_n(N),
     .                     xx_n(N),yy_n(N), h_n(N),hh_n(N),
     .                      r_n(N), d_n(N),
     .                      i_n(N), j_n(N), p_n(N),
     .                      q_n(N), c_N(N), s_n(N),
     .                     uu_n(N),vv_N(n),N)
         do F = 01, _FIELDs_
            do i = 01, 20
               if (CHAROUT_F(F)(i:i).ne.' ') then
                  if (C1_F(F).gt.i) C1_F(F) = i
                  if (C2_F(F).lt.i) C2_F(F) = i
                  endif
               enddo
            enddo
         enddo
      print*,' '
      print*,'--------------'
      print*,'FIND SPACING: '
      print*,'--------------'
      do F = 01, _FIELDs_
         write(*,419) F,C1_F(F),C2_F(F),CHAR_HEADER(F)
  419    format(5x,'F',i2.2,1x,'(',i2.2,':',i2.2,')',5x,23a)
         enddo
      print*,'                     '
      print*,'*********************'
      print*,'***'
      print*,'*** NLISTs: ',NLISTs
      print*,'***'
      print*,'*********************'
      print*,'                     '
      do NLIST = 1, NLISTs
         write(STRING300H,'(''#'',299('' ''))')
         FILEOUT = OUTDIR(1:LNC(OUTDIR,80))
     .             // PREFIX(1:LNC(PREFIX,80))
     .             // '.'
     .             // OUTLIST_NL(NLIST)(1:NITEMSO_NL(NLIST))
         open(97,file=FILEOUT,status='unknown')
         do i = 01, HHs
            write(97,'(249a)') HH(i)(1:LNC(HH(i),249))
            enddo
         write(97,'(''#--------------------------------------'')')
         write(97,'(''#  FILEOUT: '',a80)') FILEOUT
         write(97,'(''#--------------------------------------'')')
         write(97,'(''#'')')
         CLAST = 2
         do NITEM = 1, NITEMSO_NL(NLIST)
            F_NITEM(NITEM) = 0
            do F = 1, _FIELDs_
               if (char_header(F)(1:1).eq.
     .             OUTLIST_NL(NLIST)(NITEM:NITEM))
     .             F_NITEM(NITEM) = F
               enddo
            F = F_NITEM(NITEM)
            if (F.eq.0) then
               print*,'          '
               print*,'ERROR IN OUTPUT LIST...',OUTLIST_NL(NLIST)
               print*,'   NITEM: ',NITEM
               print*,'        : ',OUTLIST_NL(NLIST)(NITEM:NITEM)
               print*,'        : '
               print*,'          '
               print*,'FIELD NOT FOUND'
               stop
               endif
            N1_NITEM(NITEM) = CLAST
            N2_NITEM(NITEM) = CLAST + (C2_F(F)-C1_F(F))
            write(97,297) NITEM,CHAR_HEADER(F),
     .                    N1_NITEM(NITEM),
     .                    N2_NITEM(NITEM),F
  297       format('# (',i2.2,')',2x,a23,
     .                            4x,'c',i3.3,'-c',i2.2,
     .                            4x,'F',i3.3)
            CLAST = N2_NITEM(NITEM) + 2
            do i = N1_NITEM(NITEM), N2_NITEM(NITEM)
               STRING300H(i:i) = '.'
               enddo
            i = (N1_NITEM(NITEM)+N2_NITEM(NITEM)+1)/2
            STRING300H(i:i) = OUTLIST_NL(NLIST)(NITEM:NITEM)
            enddo!NITEM
         write(97,'(''#'')')
         if (CLAST.lt.2) CLAST = 2
         write(97,'(300a)') STRING300H(1:clast)
         write(97,'(''#'')')
         do N = 1, Ns
           !N  = NN
           !if (DOMSORT) N  = om_n(NN)
            call fill_charout(CHAROUT_F,
     .                         u_n(N), v_n(N),mm_n(N),
     .                         x_n(N), y_n(N), m_n(N), k_n(N),
     .                        xx_n(N),yy_n(N), h_n(N),hh_n(N),
     .                         r_n(N), d_n(N),
     .                         i_n(N), j_n(N), p_n(N),
     .                         q_n(N), c_N(N), s_n(N),
     .                        uu_n(N),vv_N(N),N)
            write(STRING300,'(300('' ''))')
            do NITEM = 1, NITEMSO_NL(NLIST)
               F  =  F_NITEM(NITEM)
               N1 = N1_NITEM(NITEM)
               N2 = N2_NITEM(NITEM)
               STRING300(N1:N2) = CHAROUT_F(F)(c1_F(F):c2_F(F))
               enddo
            write(97,'(300a)')
     .         STRING300(1:N2_NITEM(NITEMSO_NL(NLIST)))
            enddo
         write(97,'(''#'')')
         write(97,'(300a)') STRING300H(1:clast)
         write(97,'(''#'')')
         close(97)
         enddo!NLIST
 
      return
      end
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/HST1PASS/ROUTINES/output_regs.f"
c****                                         
c*********************************************
                                              
 
 
      subroutine output_regs(NREGs,NITEMSO_NR,OUTLIST_NR,
     .                       OUTDIR,PREFIX,HIFLAG,
     .                       HH, HHs,
     .                       Ns,  u_n,  v_n, mm_n,
     .                            x_n,  y_n,  m_n,  k_n,
     .                           xx_n, yy_n,
     .                           uu_n, vv_n,
     .                            h_n, hh_n,
     .                            r_n,  d_n,
     .                            i_n,  j_n,  p_n,
     .                            q_n,  c_n)
      implicit none
 
      integer       NREGs
      character*080 OUTLIST_NR(9)  ! the suffix of the output file
      integer       NITEMSO_NR(9)  ! the number of elements to output
 
      character*080 OUTDIR
      character*080 PREFIX
      integer       HIFLAG
 
      integer H, HHo, HHs
      character*249 HH(999)   ! max 999 header pages!
 
      integer        N, Ns
      real*8         u_n(_NSTMAX_),  v_n(_NSTMAX_), mm_n(_NSTMAX_)
      real*8         x_n(_NSTMAX_),  y_n(_NSTMAX_),  m_n(_NSTMAX_)
      real*8        xx_n(_NSTMAX_), yy_n(_NSTMAX_)
      real*8        uu_n(_NSTMAX_), vv_n(_NSTMAX_)
      integer        k_n(_NSTMAX_)
      integer        h_n(_NSTMAX_)
      real          hh_n(_NSTMAX_)
      real*8         r_n(_NSTMAX_),  d_n(_NSTMAX_)
      integer        i_n(_NSTMAX_),  j_n(_NSTMAX_)
      real           p_n(_NSTMAX_)
      real           q_n(_NSTMAX_),  c_n(_NSTMAX_)
c
c-------------------------------------------------
c
      integer       NREG
      real          rr
      integer       LNC
      character*080 FILEOUT
 
      if (NREGs.eq.0) return
 
 
      print*,'                     '
      print*,'*********************'
      print*,'***'
      print*,'***  NREGs: ',NREGs
      print*,'***'
      print*,'*********************'
      print*,'                     '
      do NREG = 1, NREGs
         FILEOUT = OUTDIR(1:LNC(OUTDIR,80))
     .             // PREFIX(1:LNC(PREFIX,80))
     .             // '_'
     .             // OUTLIST_NR(NREG)(1:2)
     .             // '.reg'
         open(98,file=FILEOUT,status='unknown')
         write(98,211)
         write(98,212)
         if (OUTLIST_NR(NREG).eq.'xy') write(98,'(''image'')')
         if (OUTLIST_NR(NREG).eq.'XY') write(98,'(''image'')')
         if (OUTLIST_NR(NREG).eq.'uv') write(98,'(''physical'')')
         if (OUTLIST_NR(NREG).eq.'rd') write(98,'(''fk5'')')
         print*,'    NREG: ',NREG,FILEOUT(1:40)
         do N = 1, Ns
            rr = -m_n(N)*0.50
            if (rr.lt.0.1) rr = 0.1
            if (rr.gt.9.9) rr = 9.9
            if (OUTLIST_NR(NREG).eq.'xy') then
               if (q_n(N).gt.0.00) then
                   write(98,216) x_n(N),y_n(N),rr
                  else
                   write(98,217) x_n(N),y_n(N),rr
                  endif
               endif
            if (OUTLIST_NR(NREG).eq.'XY') then
               if (q_n(N).gt.0.00) then
                   write(98,216) xx_n(N),yy_n(N),rr
                  else
                   write(98,217) xx_n(N),yy_n(N),rr
                  endif
               endif
            if (OUTLIST_NR(NREG).eq.'uv') then
               if (q_n(N).gt.0.00) then
                   write(98,216) u_n(N),v_n(N),rr
                  else
                   write(98,217) u_n(N),v_n(N),rr
                  endif
               endif
            if (OUTLIST_NR(NREG).eq.'rd') then
               if (q_n(N).gt.0.00) then
                   write(98,214) r_n(N),d_n(N)
                  else
                   write(98,215) r_n(N),d_n(N)
                  endif
               endif
            enddo!Ns
         enddo!NREGs
 
  211 format('# Region file format: DS9 version 4.1')
  212 format('global color=green dashlist=8 3 width=1',
     .       ' font="helvetica 10 normal" select=1 highlite=1',
     .       ' dash=0 fixed=0 edit=1 move=1 delete=1 include=1',
     .       ' source=1')
  214 format('circle(',f11.7,',',f11.7,',0.05")')                     ! UNSAT rd
  215 format('circle(',f11.7,',',f11.7,',0.05") # width=4 color=red') !   SAT rd
  216 format('circle(',f9.2,',',f9.2,',',f5.2,')')                     ! UNSAT xy,XY,uv
  217 format('circle(',f9.2,',',f9.2,',',f5.2,') # width=4 color=red') !   SAT xy,XY,uv
 
      return
      end
 
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/FITSIO/GEN/readfits_i2e.f"
c****                                         
c*********************************************
                                              
 
c-----------------------------------------------------------
c
c read in an i2 image with extensions
c
      subroutine readfits_i2e(FILEI,pix,NDIMX,NDIMY,NEXTENU)
      implicit none
 
      character*(*) FILEI
      integer   NDIMX,NDIMY
      integer   NEXTENU
      integer*2 pix(NDIMX,NDIMY)
 
      character*199 FILEU
      character*070 INFO(10)
c     common / fitsinfo / INFO
 
      integer naxes
      integer laxis(3)
      integer NXF
      integer NYF
 
      character*8  field
      character*40 stream
 
      integer nbyte0
      integer nbyteE
      integer nbyte1
      integer nbyte2
      integer nbper
      integer i,ios, k
      integer j
 
      character*2880 buffc
      byte buffb(2880)
      equivalence (buffc,buffb)
 
      integer ifirst, i1, i2
 
      integer np1, np2, npt
      integer nextend
      integer nread
      real bscale, bzero
      integer bitpix
 
      character*70 HDR(25)
c     common/HDR/HDR
 
      logical DIAG
      data DIAG /.false./
      integer NEND
      integer ii
 
      FILEU = FILEI
      do i = 195,2,-1
         if (FILEI(i:i+4).eq.'.fits') then
            FILEU = FILEI(1:i+4)
            do ii = i+5, 80
               FILEU(ii:ii) = ' '
               enddo
            endif
         enddo
 
      if (DIAG) then
         print*,'enter readfits_i2e...'
         write(*,'(''FILEI: '',80a)') FILEI
         write(*,'(''FILEU: '',80a)') FILEU
         endif
 
      open(10,file=FILEU,status='old',
     .     err=900,recl=2880,form='UNFORMATTED',
     .     access='DIRECT')
 
      if (DIAG) print*,'...opened'
 
      naxes = -1
      laxis(1) = 1
      laxis(2) = 1
      laxis(3) = 1
      nextend = 0
 
      do i = 1, 10
         INFO(i) = ' '
         enddo
 
      BSCALE = 1.0
      BZERO  = 0.0
 
      NEXTEND = 0
      NEND = -1
      i = 0
      nread = 0
 100  continue
      i = i + 1
      if (DIAG) print*,'READREC: ',i
      read(10,rec=i,iostat=ios) buffc
      do k = 0, 35, 1
         if (DIAG) write(*,'(i8,1x,i4,1x,a80)')
     .                   i,k,buffc(k*80+1:k*80+80)
         field  = buffc(k*80+01:k*80+08)
         stream = buffc(k*80+11:k*80+51)
         if (field.eq.'NAXIS   ') read(stream,*) naxes
         if (field.eq.'NAXIS1  ') read(stream,*) laxis(1)
         if (field.eq.'NAXIS2  ') read(stream,*) laxis(2)
         if (field.eq.'NAXIS3  ') read(stream,*) laxis(3)
         if (field.eq.'NEXTEND ') read(stream,*) nextend
         if (field.eq.'BITPIX  ') read(stream,*) bitpix
         if (field.eq.'BSCALE  ') read(stream,*) bscale
         if (field.eq.'BZERO   ') read(stream,*) bzero
 
         if (field.eq.'EXPTIME ') INFO(1) = stream
         if (field.eq.'FILTNAM1') INFO(2) = stream
         if (field.eq.'FILENAME') INFO(3) = stream
         if (field.eq.'DATE-OBS') INFO(4) = stream
         if (field.eq.'TIME-OBS') INFO(5) = stream
         if (field.eq.'DEC_TARG') INFO(6) = stream
         if (field.eq.'RA_TARG ') INFO(7) = stream
         if (field.eq.'DEC_DEG ') INFO(6) = stream
         if (field.eq.'RA_DEG  ') INFO(7) = stream
         if (field.eq.'PA_V3   ') INFO(8) = stream
         if (field.eq.'PROPOSID') INFO(9) = stream
 
         if (field.eq.'CRPIX1  ') HDR(01) = stream
         if (field.eq.'CRPIX2  ') HDR(02) = stream
         if (field.eq.'CRVAL1  ') HDR(03) = stream
         if (field.eq.'CRVAL2  ') HDR(04) = stream
         if (field.eq.'CTYPE1  ') HDR(05) = stream
         if (field.eq.'CTYPE2  ') HDR(06) = stream
         if (field.eq.'CD1_1   ') HDR(07) = stream
         if (field.eq.'CD1_2   ') HDR(08) = stream
         if (field.eq.'CD2_1   ') HDR(09) = stream
         if (field.eq.'CD2_2   ') HDR(10) = stream
         if (field.eq.'ORIENTAT') HDR(11) = stream
         if (field.eq.'PA_APER ') HDR(12) = stream
         if (field.eq.'PA_V3   ') HDR(13) = stream
         if (field.eq.'DATE-OBS') HDR(14) = stream
         if (field.eq.'TIME-OBS') HDR(15) = stream
         if (field.eq.'EXPTIME ') HDR(16) = stream
         if (field.eq.'ROOTNAME') HDR(17) = stream
         if (field.eq.'TARGNAME') HDR(18) = stream
         if (field.eq.'RA_TARG ') HDR(19) = stream
         if (field.eq.'DEC_TARG') HDR(20) = stream
         if (field.eq.'RA_DEG  ') HDR(19) = stream
         if (field.eq.'DEC_DEG ') HDR(20) = stream
         if (field.eq.'PROPOSID') HDR(21) = stream
         if (field.eq.'FILTER1 ') HDR(22) = stream
         if (field.eq.'FILTER2 ') HDR(23) = stream
         if (field.eq.'VAFACTOR') HDR(24) = stream
         if (field.eq.'CCDGAIN ') HDR(25) = stream
         if (field.eq.'END     ') then
            if (NEXTENU.gt.NEXTEND.and.NEXTENU.gt.1) then
               print*,'                     '
               write(*,'(''readfits_i2e: '',80a)') FILEI
               print*,'     NEXTENU.lt.NEXTEND...'
               print*,'     ---> NEXTEND: ',NEXTEND
               print*,'     ---> NEXTENU: ',NEXTENU
               print*,'                          '
               stop
               endif
            NEND = NEND + 1
            if (NEND.ge.1) goto 101
            endif
         enddo
      goto 100
 101  continue
 
      nread = nread + 1
      if (DIAG) then
         print*,' '
         print*,'----------------------------------------'
         print*,'   NREAD: ',nread
         print*,' NEXTEND: ',nextend
         print*,' NEXTENU: ',nextenu
         print*,'   NAXIS: ',naxes
         print*,'   LAXIS: ',laxis(1),laxis(2),laxis(3)
         print*,'  BITPIX: ',bitpix
         print*,'  BSCALE: ',bscale
         print*,'   BZERO: ',bzero
         print*,'   NDIMX: ',NDIMX
         print*,'   NDIMY: ',NDIMY
         print*,' '
         endif
 
      ifirst = i+1
      i1 = i
      i2 = i
 
      NXF = laxis(1)
      NYF = laxis(2)
 
      nbper  = 2*laxis(1)*laxis(2)
      if (BITPIX.eq.  8) nbper = 1*laxis(1)*laxis(2)
      if (BITPIX.eq. 32) nbper = 4*laxis(1)*laxis(2)
      if (BITPIX.eq.-32) nbper = 4*laxis(1)*laxis(2)
      if (BITPIX.eq.-64) nbper = 8*laxis(1)*laxis(2)
 
      npt    =   laxis(1)*laxis(2)
      npt    =   min(NDIMX*NDIMY,npt)
      nbyte1 = 1
      nbyte2 = nbper
      i1 = i+1 + nbyte1/2880
      i2 = i+1 + nbyte2/2880
 
      if (DIAG) then
          print*,'  i : ',i
          print*,'  i1: ',i1
          print*,'  i2: ',i2
          print*,' npt: ',npt
          print*,'NEND: ',NEND,NEXTENU,BITPIX
          endif
 
 
c     print*,'NEND: ',NEND,NEXTENU,BITPIX
      if (NEND.ne.NEXTENU) then
          if (naxes.ne.0) i = i2
          goto 100
          endif
 
      if (DIAG) then
          print*,' '
          print*,'OK!  TIME TO READ...'
          print*,'      i1: ',i1
          print*,'      i2: ',i2
          print*,'     npt: ',npt
          print*,'   NDIMX: ',NDIMX
          print*,'   NDIMY: ',NDIMY
          print*,' '
          endif
 
      if (BITPIX.ne.16) then
         print*,'BITPIX: ',BITPIX
         print*,'prob... should be 16 for readfits_i2e...'
         stop
         endif
 
      do i = i1, i2, 1
         if (DIAG) write(*,1115) i
         read(10,rec=i,iostat=ios) buffc
         nbyte0 = (i-ifirst)*2880+   1
         nbyteE = (i-ifirst)*2880+2880
         np1 = (nbyte0-nbyte1)/2 + 1
         np2 = (nbyteE-nbyte1)/2 + 1
         if (DIAG) write(*,1115) i,np1,np2,npt
         call buff2pix_i2e(buffb,pix,np1,npt)
         if (DIAG) write(*,1115) i,np1,np2,npt,i/laxis(1)
 1115    format(1x,i8,1x,i10,1x,i10,1x,i10,1x,i6.6)
         enddo
 
 
      close(10)
      return
 
 900  continue
      print*,'FILEU: ',FILEU
      print*,'READFITS_I2E ERROR'
      stop
 
      end
 
 
 
 
      subroutine buff2pix_i2e(buff,pix,n1,nt)
      implicit none
      byte buff(2880)
      integer*2 pix(*)
      integer n1,nt
 
      byte b(2)
      integer*2 ii
      equivalence(ii,b)
 
      integer i, npu, nbu
 
      logical islinux
 
      do i = 1, 1440
         npu = n1+i-1
         nbu = (i-1)*2
         if (.not.islinux()) then
            b(1) = buff(nbu+1)
            b(2) = buff(nbu+2)
            endif
         if (islinux()) then
            b(2) = buff(nbu+1)
            b(1) = buff(nbu+2)
            endif
         if (npu.ge.1.and.npu.le.nt) pix(npu) = ii
         enddo
 
      return
      end
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/FITSIO/GEN/readfits_i4e.f"
c****                                         
c*********************************************
                                              
 
c----------------------------------------------------------
c
c read in an i4 image with extensions
c
 
      subroutine readfits_i4e(FILEI,pix,NDIMX,NDIMY,NEXTENU)
      implicit none
 
      character*(*) FILEI
      integer   NDIMX,NDIMY
      integer   NEXTENU
      integer*4 pix(NDIMX,NDIMY)
 
      character*80 FILEU
      character*70 INFO(10)
      common / fitsinfo / INFO
 
      integer naxes
      integer laxis(3)
      integer NXF
      integer NYF
 
      character*8  field
      character*40 stream
 
      integer nbyte0
      integer nbyteE
      integer nbyte1
      integer nbyte2
      integer nbper
      integer i,ios, k
      integer j
 
      character*2880 buffc
      byte buffb(2880)
      equivalence (buffc,buffb)
 
      integer ifirst, i1, i2
 
      integer np1, np2, npt
      integer nextend
      integer nread
      real*8 bscale, bzero
      integer bitpix
 
      character*70 HDR(25)
      common/HDR/HDR
 
      logical DIAG
      data DIAG /.false./
      integer NEND
      integer ii
 
      FILEU = FILEI
      do i = 75,2,-1
         if (FILEI(i:i+4).eq.'.fits') then
            FILEU = FILEI(1:i+4)
            do ii = i+5, 80
               FILEU(ii:ii) = ' '
               enddo
            endif
         enddo
 
      if (DIAG) then
         print*,'enter readfits_i2e...'
         write(*,'(''FILEI: '',80a)') FILEI
         write(*,'(''FILEU: '',80a)') FILEU
         endif
 
      open(10,file=FILEU,status='old',
     .     err=900,recl=2880,form='UNFORMATTED',
     .     access='DIRECT')
 
      if (DIAG) print*,'...opened'
 
      naxes = -1
      laxis(1) = 0
      laxis(2) = 1
      laxis(3) = 1
      nextend = 0
 
      do i = 1, 10
         INFO(i) = ' '
         enddo
 
      BSCALE = 1.0
      BZERO  = 0.0
 
      NEXTEND = 999
      NEND = -1
      i = 0
      nread = 0
 100  continue
      i = i + 1
      if (DIAG) print*,'READREC: ',i
      read(10,rec=i,iostat=ios) buffc
      do k = 0, 35, 1
         if (DIAG) write(*,'(i4,1x,i4,1x,a80)')
     .                   i,k,buffc(k*80+1:k*80+80)
         field  = buffc(k*80+01:k*80+08)
         stream = buffc(k*80+11:k*80+51)
         if (field.eq.'NAXIS   ') read(stream,*) naxes
         if (field.eq.'NAXIS1  ') read(stream,*) laxis(1)
         if (field.eq.'NAXIS2  ') read(stream,*) laxis(2)
         if (field.eq.'NAXIS3  ') read(stream,*) laxis(3)
         if (field.eq.'NEXTEND ') read(stream,*) nextend
         if (field.eq.'BITPIX  ') read(stream,*) bitpix
         if (field.eq.'BSCALE  ') read(stream,*) bscale
         if (field.eq.'BZERO   ') read(stream,*) bzero
 
         if (field.eq.'EXPTIME ') INFO(1) = stream
         if (field.eq.'FILTNAM1') INFO(2) = stream
         if (field.eq.'FILENAME') INFO(3) = stream
         if (field.eq.'DATE-OBS') INFO(4) = stream
         if (field.eq.'TIME-OBS') INFO(5) = stream
         if (field.eq.'DEC_TARG') INFO(6) = stream
         if (field.eq.'RA_TARG ') INFO(7) = stream
         if (field.eq.'DEC_DEG ') INFO(6) = stream
         if (field.eq.'RA_DEG  ') INFO(7) = stream
         if (field.eq.'PA_V3   ') INFO(8) = stream
         if (field.eq.'PROPOSID') INFO(9) = stream
 
         if (field.eq.'CRPIX1  ') HDR(01) = stream
         if (field.eq.'CRPIX2  ') HDR(02) = stream
         if (field.eq.'CRVAL1  ') HDR(03) = stream
         if (field.eq.'CRVAL2  ') HDR(04) = stream
         if (field.eq.'CTYPE1  ') HDR(05) = stream
         if (field.eq.'CTYPE2  ') HDR(06) = stream
         if (field.eq.'CD1_1   ') HDR(07) = stream
         if (field.eq.'CD1_2   ') HDR(08) = stream
         if (field.eq.'CD2_1   ') HDR(09) = stream
         if (field.eq.'CD2_2   ') HDR(10) = stream
         if (field.eq.'ORIENTAT') HDR(11) = stream
         if (field.eq.'PA_APER ') HDR(12) = stream
         if (field.eq.'PA_V3   ') HDR(13) = stream
         if (field.eq.'DATE-OBS') HDR(14) = stream
         if (field.eq.'TIME-OBS') HDR(15) = stream
         if (field.eq.'EXPTIME ') HDR(16) = stream
         if (field.eq.'ROOTNAME') HDR(17) = stream
         if (field.eq.'TARGNAME') HDR(18) = stream
         if (field.eq.'RA_TARG ') HDR(19) = stream
         if (field.eq.'DEC_TARG') HDR(20) = stream
         if (field.eq.'RA_DEG  ') HDR(19) = stream
         if (field.eq.'DEC_DEG ') HDR(20) = stream
         if (field.eq.'PROPOSID') HDR(21) = stream
         if (field.eq.'FILTER1 ') HDR(22) = stream
         if (field.eq.'FILTER2 ') HDR(23) = stream
         if (field.eq.'VAFACTOR') HDR(24) = stream
         if (field.eq.'CCDGAIN ') HDR(25) = stream
         if (field.eq.'END     ') then
            if (NEXTENU.gt.NEXTEND) then
               print*,'                     '
               write(*,'(''readfits_i2e: '',80a)') FILEI
               print*,'     NEXTENU.lt.NEXTEND...'
               print*,'     ---> NEXTEND: ',NEXTEND
               print*,'     ---> NEXTENU: ',NEXTENU
               print*,'                          '
               stop
               endif
            NEND = NEND + 1
            if (NEND.ge.1) goto 101
            endif
         enddo
      goto 100
 101  continue
 
      nread = nread + 1
      if (DIAG) then
         print*,' '
         print*,'----------------------------------------'
         print*,'   NREAD: ',nread
         print*,' NEXTEND: ',nextend
         print*,' NEXTENU: ',nextenu
         print*,'   NAXIS: ',naxes
         print*,'   LAXIS: ',laxis(1),laxis(2),laxis(3)
         print*,'  BITPIX: ',bitpix
         print*,'  BSCALE: ',bscale
         print*,'   BZERO: ',bzero
         print*,'   NDIMX: ',NDIMX
         print*,'   NDIMY: ',NDIMY
         print*,' '
         endif
 
      ifirst = i+1
      i1 = i
      i2 = i
 
      NXF = laxis(1)
      NYF = laxis(2)
 
      nbper  = 4*laxis(1)*laxis(2)
      npt    =   laxis(1)*laxis(2)
      nbyte1 = 1
      nbyte2 = nbper
      i1 = i+1 + nbyte1/2880
      i2 = i+1 + nbyte2/2880
 
c     print*,'NEND: ',NEND,NEXTENU,BITPIX
      if (NEND.ne.NEXTENU) goto 100
 
      if (NAXES.eq.0) return
 
      if (BITPIX.ne.32) then
         print*,'readfits_i4e: '
         print*,'BITPIX: ',BITPIX
         print*,'prob'
         stop
         endif
 
      do i = i1, i2, 1
         read(10,rec=i,iostat=ios) buffc
         nbyte0 = (i-ifirst)*2880+   1
         nbyteE = (i-ifirst)*2880+2880
         np1 = (nbyte0-nbyte1)/4 + 1
         np2 = (nbyteE-nbyte1)/4 + 1
         call buff2pix_i4e(buffb,pix(1,1),np1,npt)
         if (DIAG) write(*,1115) i,np1,np2,npt,i/laxis(1)
 1115    format(1x,i8,1x,i10,1x,i10,1x,i10,1x,i6.6)
         enddo
 
      if (BZERO.ne.0) then
          print*,'   BZERO: ',BZERO
          print*,'  BSCALE: ',BSCALE
          print*,'pix(1712,1408): ',pix(1712,1408)
          print*,'pix(1712,1418): ',pix(1712,1418)
          print*,'pix(1614,1485): ',pix(1614,1485)
          endif
 
      close(10)
 
      do i = 1, NDIMX
      do j = 1, NDIMY
         pix(i,j) = BZERO + BSCALE*pix(i,j)
         enddo
         enddo
 
      if (BZERO.ne.0) then
          print*,'   BZERO: ',BZERO
          print*,'  BSCALE: ',BSCALE
          print*,'pix(1712,1408): ',pix(1712,1408)
          print*,'pix(1712,1418): ',pix(1712,1418)
          print*,'pix(1614,1485): ',pix(1614,1485)
          endif
 
      close(10)
      return
 
 900  continue
      print*,'readfits_i4e ERROR'
      stop
 
      end
 
 
 
 
      subroutine buff2pix_i4e(buff,pix,n1,nt)
      implicit none
      byte buff(2880)
      integer*4 pix(*)
      integer n1,nt
 
      byte b(4)
      integer ii
      equivalence(ii,b)
 
      integer i, npu, nbu
      logical islinux
 
      do i = 0001, 0720
         npu = n1+i-1
         nbu = (i-1)*4
         if (.not.(islinux())) then
             b(1) = buff(nbu+1)
             b(2) = buff(nbu+2)
             b(3) = buff(nbu+3)
             b(4) = buff(nbu+4)
            else
             b(4) = buff(nbu+1)
             b(3) = buff(nbu+2)
             b(2) = buff(nbu+3)
             b(1) = buff(nbu+4)
             endif
         if (npu.ge.1.and.npu.le.nt) pix(npu) = ii
         enddo
 
      return
      end
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/FITSIO/GEN/readfits_r4_4D.f"
c****                                         
c*********************************************
                                              
 
 
c------------------------------------------------------
c
c
c
      subroutine readfits_r4_4D(FILE,pix,N1,N2,N3,N4)
      implicit none
 
      character*80 FILE
      integer N1, N2, N3, N4
      real*4  pix(N1,N2,N3,N4)
 
      integer naxes
      integer laxis(4)
 
      character*8  field
      character*20 stream
 
      real    pixu
      integer nbyte0
      integer nbyteE
      integer nbyte1
      integer nbyte2
      integer nbper
      integer i,ios, k
      integer ii, jj, kk, ll
      integer n
      integer N1U, N2U, N3U, N4U
 
      character*2880 buffc
      byte buffb(2880)
      equivalence (buffc,buffb)
 
      real*4 rbuff(720)
 
      integer ifirst, i1, i2
      integer j
 
      integer np1, np2, npt
      integer nextend
      integer nread
      real bscale, bzero
      integer bitpix
 
      integer r2i
 
      logical LINUX
      data LINUX/.true./
      logical DIAG
      data DIAG /.false./
 
 
      character*80 FILEU
 
      FILEU = FILE
      do i = 75,2,-1
         if (FILE(i:i+4).eq.'.fits') FILEU = FILE(1:i+4)
         enddo
 
 
      open(10,file=FILEU,status='old',
     .     err=900,recl=2880,form='UNFORMATTED',
     .     access='DIRECT')
 
      if (DIAG) print*,'...opened'
 
      bscale = 1
      bzero  = 0
 
      naxes = -1
      laxis(1) = 1
      laxis(2) = 1
      laxis(3) = 1
      laxis(4) = 1
      nextend = 0
 
      i = 0
      nread = 0
 100  continue
      i = i + 1
      read(10,rec=i,iostat=ios) buffc
      if (DIAG) print*,'READREC: ',i
      do k = 0, 35, 1
         if (DIAG) write(*,'(i4,1x,i4,1x,a80)')
     .                   i,k,buffc(k*80+1:k*80+80)
         field  = buffc(k*80+01:k*80+08)
         stream = buffc(k*80+11:k*80+31)
         if (field.eq.'NAXIS   ') read(stream,*) naxes
         if (field.eq.'NAXIS1  ') read(stream,*) laxis(1)
         if (field.eq.'NAXIS2  ') read(stream,*) laxis(2)
         if (field.eq.'NAXIS3  ') read(stream,*) laxis(3)
         if (field.eq.'NAXIS4  ') read(stream,*) laxis(4)
         if (field.eq.'NEXTEND ') read(stream,*) nextend
         if (field.eq.'BITPIX  ') read(stream,*) bitpix
         if (field.eq.'BSCALE  ') read(stream,*) bscale
         if (field.eq.'BZERO   ') read(stream,*) bzero
         if (field.eq.'END     ') goto 101
         enddo
      goto 100
 101  continue
 
      nread = nread + 1
      if (DIAG) then
         print*,'----------------------------------------'
         print*,'  NREAD: ',nread
         print*,'NEXTEND: ',nextend
         print*,'  NAXIS: ',naxes
         print*,'  LAXIS: ',laxis(1),laxis(2),laxis(3)
         print*,' BITPIX: ',bitpix
         print*,' BSCALE: ',bscale
         print*,'  BZERO: ',bzero
         endif
 
      ifirst = i+1
      i1 = i
      i2 = i
 
      if (BITPIX.ne.-32) then
         print*,'readfits_r4...: '
         print*,'                '
         print*,'   you called a routine to read in an'
         print*,'   real*4 mage, and the image you'
         print*,'   gave it has BITPIX = ',BITPIX
         print*,'   '
         print*,'   FILEU: ',FILEU
         print*,'   '
         stop
         endif
 
      nbper = 4*laxis(1)*laxis(2)*laxis(3)*laxis(4)
      npt   =   laxis(1)*laxis(2)*laxis(3)*laxis(4)
      nbyte1 = 1
      nbyte2 = nbper
      i1 = i+1 + nbyte1/2880
      i2 = i+1 + nbyte2/2880
 
      if (laxis(1).ne.N1.or.
     .    laxis(2).ne.N2.or.
     .    laxis(3).ne.N3.or.
     .    laxis(4).ne.N4) then
         write(*,'(''readfits_r4_4D: '',80a)') FILE
         write(*,'(''readfits_r4_4D: '',80a)') FILEU
         print*,' '
         print*,'  laxis1: ',laxis(1)
         print*,'      N1: ',N1
         print*,' '
         print*,'  laxis2: ',laxis(2)
         print*,'      N2: ',N2
         print*,' '
         print*,'  laxis3: ',laxis(3)
         print*,'      N3: ',N3
         print*,' '
         print*,'  laxis4: ',laxis(4)
         print*,'      N4: ',N4
         print*,' '
         stop
         endif
 
      N1U = laxis(1)
      N2U = laxis(2)
      N3U = laxis(3)
      N4U = laxis(4)
 
      if (DIAG) then
         print*,'     N1: ',N1
         print*,'     N2: ',N2
         print*,'     N3: ',N3
         print*,'     N4: ',N4
         print*,'  NBPER: ',nbper
         print*,'  NBYT1: ',nbyte1
         print*,'  NBYT2: ',nbyte2
         print*,' IFIRST: ',ifirst
         print*,'     I1: ',i1
         print*,'     I2: ',i2
         print*,'    NPT: ',NPT
         endif
 
      do i = i1, i2, 1
         read(10,rec=i,iostat=ios) buffc
         nbyte0 = (i-ifirst)*2880+   1
         nbyteE = (i-ifirst)*2880+2880
         np1 = (nbyte0-nbyte1)/4 + 1
         np2 = (nbyteE-nbyte1)/4 + 1
         np2 = min(np2,npt)
         call buff2pix_r4_4D(buffb,rbuff,0001,0720)
         do n = np1, np2, 1
            ll = n/(N1U*N2U*N3U)
            kk = (n-ll*N1U*N2U*N3U)/N1U/N2U
            jj = (n-ll*N1U*N2U*N3U-kk*N1U*N2U)/N1U
            ii =  n-ll*N1U*N2U*N3U-kk*N1U*N2U-jj*N1U
            pixu = rbuff(n-np1+1)*bscale+bzero
            pix(ii,jj+1,kk+1,ll+1) = pixu
            enddo
         enddo
 
      if (DIAG) then
         print*,'  NBPER: ',nbper
         print*,'  NBYT1: ',nbyte1
         print*,'  NBYT2: ',nbyte2
         print*,' IFIRST: ',ifirst
         print*,'     I1: ',i1
         print*,'     I2: ',i2
         print*,'    NPT: ',NPT
         endif
 
 
      close(10)
      return
 
 900  continue
      print*,' '
      print*,'subroutine readfits_r4_4D'
      print*,' '
      print*,'  FILE OPEN ERROR'
      print*,'  FILE : ',FILE
      print*,'  FILEU: ',FILEU
      print*,' '
      stop
 
      end
 
 
 
      subroutine buff2pix_r4_4D(buff,pix,n1,nt)
      implicit none
      byte    buff(2880)
      real*4  pix(720)
      integer n1,nt
 
      byte   b(4)
      real*4 r
      equivalence(r,b)
 
      integer i, npu, nbu
      logical islinux
 
      do i = 0001, 0720
         npu = n1+i-1
         nbu = (i-1)*4
         if (.not.islinux()) then
            b(1) = buff(nbu+1)
            b(2) = buff(nbu+2)
            b(3) = buff(nbu+3)
            b(4) = buff(nbu+4)
            endif
         if (islinux()) then
            b(4) = buff(nbu+1)
            b(3) = buff(nbu+2)
            b(2) = buff(nbu+3)
            b(1) = buff(nbu+4)
            endif
         if (npu.ge.1.and.npu.le.nt) pix(npu) = r
         enddo
 
      return
      end
 
 
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/FITSIO/GEN/rdate_header.f"
c****                                         
c*********************************************
                                              
 
 
      real*8 function rdate_header(FILENAME)
      implicit none
 
      integer i
      character filename*80
      integer NCHAR
 
      integer iargc, nargc
 
      integer iEXPT
      real    rEXPT
      integer iPA
      real    rPA
      integer iPID
      character*6 FILTSTR
      character*9 ROOTSTR
      real rRA, hRA
      real rDEC, aDEC
 
      character*20 DATESTR
      character*20 TIMESTR
 
      integer iRAH, iRAM, iRAS, iFRA
      integer iDEG, iMIN, iSEC, iFDC
      character decsgn
      real    rRA0, rDC0
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
 
 
      character*11 OBJNAME
 
      integer DOY(12)
      data DOY / 000, 031, 059, 090, 120, 151,
     .           181, 212, 243, 273, 304, 334/
 
 
      character*8 CHINJVAL
 
      DATESTR  = 'DATELESS'
      TIMESTR  = 'TIMELESS'
      rRA  = 0.
      rDEC = 0.
      iPID = 0
      iPA  = 0
      rPA  = 0
      rSUNA = 0.
      rTEMP = 0.
 
      open(10,file=FILENAME,status='old',
     .  err=900,recl=2880,form='UNFORMATTED',
     .  access='DIRECT')
 
      i = 0
      nread = 0
 100  continue
      i = i + 1
      read(10,rec=i,iostat=ios) buff
      if (ios.lt.0) goto 900
      do k = 0, 35, 1
         field  = buff(k*80+01:k*80+08)
         stream = buff(k*80+11:k*80+31)
         if (field.eq.'TIME-OBS') TIMESTR = stream(01:20)
         if (field.eq.'DATE-OBS') DATESTR = stream(01:20)
         if (field.eq.'END     ') goto 101
         enddo
      goto 100
  101 close(10)
 
      if (DATESTR(06:06).eq.'-'.and.DATESTR(09:09).eq.'-')
     .                    DATESTR = DATESTR(02:13)
      if (DATESTR(07:07).eq.'-'.and.DATESTR(10:10).eq.'-')
     .                    DATESTR = DATESTR(03:14)
      if (DATESTR(08:08).eq.'-'.and.DATESTR(11:11).eq.'-')
     .                    DATESTR = DATESTR(04:15)
      if (TIMESTR(04:04).eq.':'.and.TIMESTR(07:07).eq.':')
     .                    TIMESTR = TIMESTR(02:09)
      if (TIMESTR(05:05).eq.':'.and.TIMESTR(05:05).eq.':')
     .                    TIMESTR = TIMESTR(03:10)
 
      if (DATESTR.eq.'DATELESS') return
      if (TIMESTR.eq.'TIMELESS') return
 
      if (DATESTR.ne.'DATELESS') then
         read(DATESTR(01:04),*) iYIR
         read(DATESTR(06:07),*) iMON
         if (iMON.lt.01.or.iMON.gt.12) then
            print*,'---> iMON: ',iMON
            print*,'---> DATE: ',DATESTR
            stop
            endif
         read(DATESTR(09:10),*) iDAY
         endif
 
      if (TIMESTR.ne.'TIMELESS') then
         read(TIMESTR(1:2),*) iHRS
         read(TIMESTR(4:5),*) iMIN
         read(TIMESTR(7:8),*) iSEC
         endif
 
 
      rDAT = iYIR + (DOY(iMON)+iDAY)/365.0
      rDAT = rDAT + (iHRS + iMIN/60. + iSEC/60./60.)/8766.
 
      rdate_header = rDAT
 
      return
 
  900 continue
      print*,'               '
      print*,'rdate_header...'
      print*,'   PROBLEM OPENING FILE: ',FILENAME
      print*,'               '
      stop
      end
 
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/FITSIO/WFC3IR/read_wfc3ir_flt_full.f"
c****                                         
c*********************************************
                                              
 
 
      subroutine read_wfc3ir_flt_full_simpl(FILENAME,pixr)
      implicit none
 
      character*80 FILENAME
      real      pixr(1014,1014)
 
      real*8    CRPIX1, CRPIX2
      real*8    CRVAL1, CRVAL2
      real*8    CD1_1,  CD1_2
      real*8    CD2_1,  CD2_2
 
      integer   LOFLAG, HIFLAG
      real*8    BDRY_XR(4,4)
      real*8    BDRY_YR(4,4)
 
      character*05 FILT
      real*4       EXPT
      real*4       RDAT
      integer      PROP
 
      call read_wfc3ir_flt_full(FILENAME,pixr,
     .                          CRPIX1, CRPIX2,
     .                          CRVAL1, CRVAL2,
     .                          CD1_1,  CD1_2,
     .                          CD2_1,  CD2_2,
     .                          LOFLAG, HIFLAG,
     .                          BDRY_XR,BDRY_YR,
     .                          FILT,EXPT,RDAT,PROP)
 
      return
      end
 
 
      subroutine read_wfc3ir_flt_full(FILENAME,pixr,
     .                                CRPIX1, CRPIX2,
     .                                CRVAL1, CRVAL2,
     .                                CD1_1,  CD1_2,
     .                                CD2_1,  CD2_2,
     .                                LOFLAG, HIFLAG,
     .                                BDRY_XR,BDRY_YR,
     .                                FILT,EXPT,RDAT,PROP)
      implicit none
 
      character*80 FILENAME
      real      pixr(1014,1014)
 
      real*8    CRPIX1, CRPIX2
      real*8    CRVAL1, CRVAL2
      real*8    CD1_1,  CD1_2
      real*8    CD2_1,  CD2_2
 
      integer   LOFLAG, HIFLAG
      real*8    BDRY_XR(4,4)
      real*8    BDRY_YR(4,4)
 
      character*05 FILT
      real*4       EXPT
      real*4       RDAT
      integer      PROP
 
 
c
c-----------------------------------
c
 
      real      pix(1014,1014), pmax
      integer*2 pox(1014,1014)
      integer   pux(1014,1014)
      integer  i, j
      integer ii,jj
      real xsum, ysum, psum, msky
      real xbar, ybar, mag
      real mbar_sky
      character*70 INFO(10)
      common / fitsinfo / INFO
 
      real EXPTIME
      integer k
      real pval
      integer ival
 
      real      pixi(1014,1014)
      real      pixh(1014,1014)
      integer*2 pixo(1014,1014)
      integer*2 pixi2(1014,1014)
 
      real*4,    dimension(:,:), allocatable ::  pixi_sub
      integer*2, dimension(:,:), allocatable :: pixi2_sub
 
 
      real rsum
      real dd
      integer Ls
      real    plist(9)
      integer NIT, NFIXN, NFIXT
      integer*2 i2
 
      character*20 STREAM
      character*20 STRING_APER
      integer i4_query_hdre
      real*4  r4_query_hdre
      real*8  r8_query_hdre
      real*8  rdate_header
      real*8  PA_APER
      real*8  PA_V3
      real*8  COSPA, SINPA
 
c
c--------------------------------------------------
c
 
      do i = 0001, 1014
      do j = 0001, 1014
         pixi(i,j)  = 0.
         pixi2(i,j) = 0
         enddo
         enddo
 
      call query_hdre_r8(FILENAME,'CRPIX1  ',CRPIX1,-1)
      call query_hdre_r8(FILENAME,'CRPIX2  ',CRPIX2,-1)
      call query_hdre_r8(FILENAME,'CRVAL1  ',CRVAL1,-1)
      call query_hdre_r8(FILENAME,'CRVAL2  ',CRVAL2,-1)
c     call query_hdre_r8(FILENAME,'CD1_1   ',CD1_1,-1)
c     call query_hdre_r8(FILENAME,'CD1_2   ',CD1_2,-1)
c     call query_hdre_r8(FILENAME,'CD2_1   ',CD2_1,-1)
c     call query_hdre_r8(FILENAME,'CD2_2   ',CD2_2,-1)
 
      call query_hdre(FILENAME,'APERTURE',STRING_APER,-1)
 
      if (STRING_APER(2:9).eq.'IRSUB256') then
          print*,'APERTURE: ',STRING_APER
          allocate( pixi_sub(256,256))
          allocate(pixi2_sub(256,256))
          call readfits_r4e(FILENAME, pixi_sub,256,256,1)
          call readfits_i2e(FILENAME,pixi2_sub,256,256,3)
          do i = 001, 256
          do j = 001, 256
              pixi(i+379,j+379) =  pixi_sub(i,j)
             pixi2(i+379,j+379) = pixi2_sub(i,j)
             enddo
             enddo
          CRPIX1 = CRPIX1 + 379
          CRPIX2 = CRPIX2 + 379
 
          if (.false.) then
              print*,' '
              print*,'IRSUB256...'
              print*,' '
              print*,'    pixi_sub(0217,0201): ', pixi_sub(0217,0201)
              print*,'   pixi2_sub(0217,0201): ',pixi2_sub(0217,0201)
              print*,' '
              call writfits_r4('pixi1_sub.fits',pixi_sub, 256,256)
              call writfits_i2('pixi3_sub.fits',pixi2_sub,256,256)
              stop
              endif
 
          goto 3
          endif
 
      call readfits_r4e(FILENAME,pixi ,1014,1014,1)
      call readfits_i2e(FILENAME,pixi2,1014,1014,3)
 
    3 continue
 
 
      PA_APER = r8_query_hdre(FILENAME,'PA_APER ',-1)
      PA_V3   = PA_APER + 0.1256971
 
      COSPA = cos(PA_V3*3.14159/180.0)
      SINPA = sin(PA_V3*3.14159/180.0)
 
      CD1_1 = -COSPA*0.120978/60/60
      CD1_2 =  SINPA*0.120978/60/60
      CD2_1 =  SINPA*0.120978/60/60
      CD2_2 =  COSPA*0.120978/60/60
 
      PROP = i4_query_hdre(FILENAME,'PROPOSID',-1)
 
c     print*,'PROP: ',PROP
 
      EXPT = r4_query_hdre(FILENAME,'EXPTIME ',-1)
      call query_hdre(FILENAME,     'FILTER  ',STREAM,-1)
 
      if (STREAM(1:1).eq.'F') FILT = STREAM(1:5)  ! allow quotes in the filter name...
      if (STREAM(2:2).eq.'F') FILT = STREAM(2:6)
 
      RDAT = rdate_header(FILENAME)
 
      LOFLAG =   -50
      HIFLAG = 30000
 
      do i = 1, 4
      do j = 1, 4
         BDRY_XR(i,j) = 0.
         BDRY_YR(i,j) = 0.
         enddo
         enddo
 
      BDRY_XR(1,1) = 0001.
      BDRY_YR(1,1) = 0001.
      BDRY_XR(2,1) = 1013.
      BDRY_YR(2,1) = 0001.
      BDRY_XR(3,1) = 1013.
      BDRY_YR(3,1) = 1013.
      BDRY_XR(4,1) = 0001.
      BDRY_YR(4,1) = 1013.
 
      do i = 0001, 1014
      do j = 0001, 1014
         if (iand(pixi2(i,j),00001).ne.0) continue             ! Reed-Solomon decoding error
         if (iand(pixi2(i,j),00002).ne.0) continue             ! data missing/replaced by fill
         if (iand(pixi2(i,j),00004).ne.0) pixi(i,j) = -500     ! bad detector pixel
         if (iand(pixi2(i,j),00008).ne.0) continue             ! deviant zero-read (bias) value
         if (iand(pixi2(i,j),00016).ne.0) continue             ! hot pixel
         if (iand(pixi2(i,j),00032).ne.0) continue             ! unstable response
         if (iand(pixi2(i,j),00256).ne.0) continue             ! full-well saturation
         if (iand(pixi2(i,j),00512).ne.0) pixi(i,j) = -500     ! bad/uncertain value (blob)
         if (iand(pixi2(i,j),01024).ne.0) continue             ! (reserved)
         if (iand(pixi2(i,j),02048).ne.0) continue             ! signal in zero-read
         if (iand(pixi2(i,j),04096).ne.0) continue             ! CR detected by MDriz
         if (iand(pixi2(i,j),08192).ne.0) continue             ! CR detected during up-the-ramp
         if (iand(pixi2(i,j),16384).ne.0) continue             ! pixel affected by ghost/crosstalk
 
 
         pixr(i,j) = pixi(i,j)
         if (pixi2(i,j).eq.00256) pixr(i,j) = HIFLAG+1 ! full-well saturation ; only if there are no other flags
         enddo
         enddo
 
      do i = 1, 1014
      do j = 1, 1014
         pixh(i,j) = pixr(i,j)
         enddo
         enddo
 
      do i = 1, 1014
      do j = 1, 1014
         rsum = 0.
         if (pixr(i,j).ge.HIFLAG) then
            do ii = max(0001,i-5), min(1014,i+5)
            do jj = max(0001,j-5), min(1014,j+5)
               dd = sqrt(1.*(i-ii)**2+(j-jj)**2)
               rsum = rsum + 1.0/(1+dd)**2*pixh(ii,jj)
               enddo
               enddo
            pixr(i,j) = pixr(i,j) + rsum
            endif
         enddo
         enddo
 
 
c------------------------------------------------
c
c blank out the death star
c
      do i = 0001, 1014
      do j = 0001, 1014
         rsum = sqrt((i-359.)**2+(j-55.)**2)
         if (rsum.lt.25) pixr(i,j) = -751
         enddo
         enddo
 
c     print*,'EXIT WFC3IR_READ_FULL_WSUB: '
c     print*,'     CRPIX: ',CRPIX1,CRPIX2
c     print*,'     CRVAL: ',CRVAL1,CRVAL2
 
      return
      end
 
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/FITSIO/ACSWFC/read_acswfc_flt_full.f"
c****                                         
c*********************************************
                                              
 
 
c--------------------------------------------------
c
c
      subroutine read_acswfc_flt_smpl(FILENAME,pix)
      implicit none
 
      character*200 FILENAME
      real          pix(4096,4096)
 
      real*8    CRPIX1, CRPIX2
      real*8    CRVAL1, CRVAL2
      real*8    CD1_1,  CD1_2
      real*8    CD2_1,  CD2_2
 
      integer   LOFLAG, HIFLAG
      real*8    BDRY_XR(4,4)
      real*8    BDRY_YR(4,4)
 
      character*05 FILT
      real*4       EXPT
      real*4       RDAT
      integer      PROP
 
      byte, dimension(:,:), allocatable :: pxq
 
      allocate(pxq(4096,4096))
 
      call read_acswfc_flt_full(FILENAME,pix,pxq,
     .                          CRPIX1, CRPIX2,
     .                          CRVAL1, CRVAL2,
     .                          CD1_1,  CD1_2,
     .                          CD2_1,  CD2_2,
     .                          LOFLAG, HIFLAG,
     .                          BDRY_XR,  BDRY_YR,
     .                          FILT,EXPT,
     .                          RDAT,PROP)
 
      deallocate(pxq)
 
      return
      end
 
 
 
c---------------------------------------------------
c
c
      subroutine read_acswfc_flt_full(FILENAME,pix,pxq,
     .                                CRPIX1, CRPIX2,
     .                                CRVAL1, CRVAL2,
     .                                CD1_1,  CD1_2,
     .                                CD2_1,  CD2_2,
     .                                LOFLAG, HIFLAG,
     .                                BDRY_XR,  BDRY_YR,
     .                                FILT,EXPT,
     .                                RDAT,PROP)
      implicit none
 
      character*200 FILENAME
      real      pix(4096,4096)
      byte      pxq(4096,4096)
 
      real*8    CRPIX1, CRPIX2
      real*8    CRVAL1, CRVAL2
      real*8    CD1_1,  CD1_2
      real*8    CD2_1,  CD2_2
 
      integer   LOFLAG, HIFLAG
      real*8    BDRY_XR(4,4)
      real*8    BDRY_YR(4,4)
 
      character*05 FILT
      real*4       EXPT
      real*4       RDAT
      integer      PROP
 
c
c----------------------------------------------------
c
 
      integer NAXIS1
      integer NAXIS2
 
      real*8 COSPA, SINPA
      real*8 PA_APER
      real*8 PA_V3
 
      character*80 FILTU
      character*80 FILT1, FILT2, FILT3, FILT4
      real*8  r8_query_hdre
      real*4  r4_query_hdre
      integer i4_query_hdre
      real*8  rdate_header
 
      integer i, j
      logical str_contains
 
      real*4,    dimension(:,:),   allocatable :: pxz
      integer*2, dimension(:,:),   allocatable :: pxd ! data quality array
 
 
      if (.false.) then
         print*,'   '
         print*,'ENTER: subroutine read_acswfc_flt_full '
         print*,'---> FILENAME: ',FILENAME
         print*,'   '
         endif
 
      NAXIS1 = r8_query_hdre(FILENAME,'NAXIS1  ',-1)
      NAXIS2 = r8_query_hdre(FILENAME,'NAXIS2  ',-1)
 
      if (NAXIS1.lt.4096.and.NAXIS2.lt.2048)
     .    stop 'only FLT ACS full frame (for now)'
 
      CRPIX1 = r8_query_hdre(FILENAME,'CRPIX1  ',-1)
      CRPIX2 = r8_query_hdre(FILENAME,'CRPIX2  ',-1)
      CRVAL1 = r8_query_hdre(FILENAME,'CRVAL1  ',-1)
      CRVAL2 = r8_query_hdre(FILENAME,'CRVAL2  ',-1)
 
      PA_APER = r8_query_hdre(FILENAME,'PA_APER ',-1)
      PA_V3   = PA_APER + 0.2181083
 
      COSPA = cos(PA_V3*3.14159/180.0)
      SINPA = sin(PA_V3*3.14159/180.0)
 
      CD1_1 = -COSPA*0.049730/60/60
      CD1_2 =  SINPA*0.049730/60/60
      CD2_1 =  SINPA*0.049730/60/60
      CD2_2 =  COSPA*0.049730/60/60
 
      PROP = i4_query_hdre(FILENAME,'PROPOSID',-1)
      EXPT = r4_query_hdre(FILENAME,'EXPTIME ',-1)
      RDAT = rdate_header(FILENAME)
 
      call query_hdre(FILENAME,'FILTER  ',FILTU,-1)
      call query_hdre(FILENAME,'FILTER1 ',FILT1,-1)
      call query_hdre(FILENAME,'FILTER2 ',FILT2,-1)
      call query_hdre(FILENAME,'FILTNAM1',FILT3,-1)
      call query_hdre(FILENAME,'FILTNAM2',FILT4,-1)
 
      FILT = FILTU
      if (FILT1(1:1).eq.'F')  FILT = FILT1(1:5)
      if (FILT2(1:1).eq.'F')  FILT = FILT2(1:5)
      if (FILT3(1:1).eq.'F')  FILT = FILT3(1:5)
      if (FILT4(1:1).eq.'F')  FILT = FILT4(1:5)
      if (FILT1(2:2).eq.'F')  FILT = FILT1(2:6)
      if (FILT2(2:2).eq.'F')  FILT = FILT2(2:6)
      if (FILT3(2:2).eq.'F')  FILT = FILT3(2:6)
      if (FILT4(2:2).eq.'F')  FILT = FILT4(2:6)
 
      LOFLAG =   -50
      HIFLAG = 76000
 
      do i = 1, 4
      do j = 1, 4
         BDRY_XR(i,j) = 0.
         BDRY_YR(i,j) = 0.
         enddo
         enddo
 
      BDRY_XR(1,1) = 0001
      BDRY_YR(1,1) = 0001
      BDRY_XR(2,1) = 4095
      BDRY_YR(2,1) = 0001
      BDRY_XR(3,1) = 4095
      BDRY_YR(3,1) = 2047
      BDRY_XR(4,1) = 0001
      BDRY_YR(4,1) = 2047
 
      BDRY_XR(1,2) = 0001
      BDRY_YR(1,2) = 2049
      BDRY_XR(2,2) = 4095
      BDRY_YR(2,2) = 2049
      BDRY_XR(3,2) = 4095
      BDRY_YR(3,2) = 4095
      BDRY_XR(4,2) = 0001
      BDRY_YR(4,2) = 4095
 
 
      if (_SKIPPX_) then
          do i = 0001, 4096
          do j = 0001, 4096
             pix(i,j) = -750
             enddo
             enddo
          goto 444
          endif
 
      if (str_contains(FILENAME,200,'_flt.fits',09).or.
     .    str_contains(FILENAME,200,'_flc.fits',09).or.
     .    str_contains(FILENAME,200,'_crj.fits',09).or.
     .    str_contains(FILENAME,200,'_crc.fits',09)) then
          call readfits_WFC(FILENAME,pix(0001,2049),4)
          call readfits_WFC(FILENAME,pix(0001,0001),1)
          allocate(pxd(4096,4096))
          print*,'---> read in dqe6: ',FILENAME
          call readfits_i2e(FILENAME,pxd(0001,2049),4096,2048,6)
          print*,'---> read in dqe3: ',FILENAME
          call readfits_i2e(FILENAME,pxd(0001,0001),4096,2048,3)
          do i = 0001, 4096
          do j = 0001, 4096
             if (and(pxd(i,j),256).gt.0) pxq(i,j) = 1
             enddo
             enddo
          deallocate(pxd)
          print*,'---> goto 444 ...'
          goto 444
          endif
 
      if (str_contains(FILENAME,200,'_WJC.fits',09).or.
     .    str_contains(FILENAME,200,'_WJX.fits',09).or.
     .    str_contains(FILENAME,200,'_WXC.fits',09).or.
     .    str_contains(FILENAME,200,'_WJ2.fits',09)) then
          call readfits_j2r(FILENAME,pix(0001,0001),4096,4096)
          goto 444
          endif
 
      if (str_contains(FILENAME,200,'_raw.fits',09)) then
          call readfits_acsraw2r(FILENAME,pix)
          goto 444
          endif
 
      if (str_contains(FILENAME,200,'_raz.fits',09).or.
     .    str_contains(FILENAME,200,'_rzc.fits',09)) then
          allocate(pxz(8288,2068))
          call readfits_r4(FILENAME,pxz,8288,2068)
          do i = 0001, 2048
          do j = 0001, 2048
             pix(i+0000,j+0000) = pxz(0*2072+24+i,j)
             pix(4097-i,j+0000) = pxz(1*2072+24+i,j)
             pix(i+0000,4097-j) = pxz(2*2072+24+i,j)
             pix(4097-i,4097-j) = pxz(3*2072+24+i,j)
             enddo
             enddo
          deallocate(pxz)
          goto 444
          endif
 
      print*,'PROB : subroutine read_acswfc_flt_full '
      print*,'       could not read in file: '
      print*,'       ',FILENAME
      stop
  444 continue
 
      do i = 0001, 4096
      do j = 2046, 2051
         pix(i,j) = -750
         enddo
         enddo
 
      return
      end
 
 
 
 
 
      subroutine readfits_WFC(FILE,pix,nimg)
      implicit none
 
      character*(*) FILE
      real pix(4096,2048)
 
      character*70 INFO(10)
      common / fitsinfo / INFO
 
      integer nimg
      integer naxes
      integer laxis(3)
 
      character*8  field
      character*20 stream
 
      integer nbyte0
      integer nbyteE
      integer nbyte1
      integer nbyte2
      integer nbper
      integer i,ios, k
 
      character*2880 buffc
      byte buffb(2880)
      equivalence (buffc,buffb)
 
      integer ifirst, i1, i2
 
      integer np1, np2, npt
      integer nextend
      integer nread
      real bscale, bzero
      integer bitpix
 
 
      logical LINUX
      data LINUX/.true./
 
      logical first_header
 
      integer NH_RECs
      character*2880 HEADER_STRING(99)
      common /HEADER_STRING_INFO/NH_RECs,HEADER_STRING
 
      character*200 FILEu
 
      FILEu = 'NONE'
      do i = 1, 195
         if (FILE(i:i).eq.' ') then
             print*,'readfits_WJC filename problem'
             print*,'   space before .fits '
             print*,'   FILE: ',FILE
             stop
             endif
         if (FILE(i:i+4).eq.'.fits') then
             FILEu = FILE(1:i+4)
             goto 3
             endif
         enddo
      if (FILE(i:i).eq.' ') then
          print*,'readfits_WJC filename problem'
          print*,'   space before .fits '
          print*,'   FILE: ',FILE
          stop
          endif
 
    3 continue
      open(10,file=FILEu,status='old',iostat=ios,
     .     err=900,recl=2880,form='UNFORMATTED',
     .     access='DIRECT')
 
      bscale = 1.0
      bzero  = 0.0
 
      naxes = -1
      laxis(1) = 1
      laxis(2) = 1
      laxis(3) = 1
      nextend = 0
 
 
      NH_RECs = 0
      first_header = .true.
      i = 0
      nread = 0
 100  continue
      i = i + 1
      read(10,rec=i,iostat=ios) buffc
      if (first_header) then
         NH_RECs = NH_RECs + 1
         HEADER_STRING(NH_RECs) = buffc
         endif
      do k = 0, 35, 1
         field  = buffc(k*80+01:k*80+08)
         stream = buffc(k*80+11:k*80+31)
         if (field.eq.'NAXIS   ') read(stream,*) naxes
         if (field.eq.'NAXIS1  ') read(stream,*) laxis(1)
         if (field.eq.'NAXIS2  ') read(stream,*) laxis(2)
         if (field.eq.'NAXIS3  ') read(stream,*) laxis(3)
         if (field.eq.'NEXTEND ') read(stream,*) nextend
         if (field.eq.'BITPIX  ') read(stream,*) bitpix
         if (field.eq.'BSCALE  ') read(stream,*) bscale
         if (field.eq.'BZERO   ') read(stream,*) bzero
 
         if (field.eq.'END     ') then
            first_header = .false.
            goto 101
            endif
         enddo
      goto 100
 101  continue
 
 
      nread = nread + 1
 
      ifirst = i+1
      i1 = i
      i2 = i
 
      if (naxes.eq.0) then    ! maybe multiple images stored as extensions
         if (nextend.eq.0) then
            print*,'THIS IS A NULL IMAGE: '
            print*,'NAXES: ',NAXES
            print*,'NEXND: ',NEXTEND
            stop
            endif
         endif
 
      if (nread.ne.nimg+1) then
         nbper  = abs(BITPIX/8)*laxis(1)*laxis(2)
         if (NAXES.eq.0) nbper = 0
         i = i + 1.0*nbper/2880 + 0.9999
         goto 100
         endif
 
      if (laxis(1).ne.4096.or.
     .    laxis(2).ne.2048) then
          print*,'  laxis1: ',laxis(1)
          print*,'  laxis2: ',laxis(2)
          print*,'    4096: ',4096
          print*,'    2048: ',2048
          stop
          endif
 
      if (naxes.eq.2) then   ! nimg is irrelevant; ignore
         nbper  = 4*laxis(1)*laxis(2)
         npt    = laxis(1)*laxis(2)
         nbyte1 = 1
         nbyte2 = nbper
         i1 = i+1 + nbyte1/2880
         i2 = i+1 + nbyte2/2880
         endif
 
      if (naxes.eq.3) then
         nbper  = 4*laxis(1)*laxis(2)
         nbyte1 = 1 + nbper*(nimg-1)
         nbyte2 =     nbper*(nimg  )
         i1 = i+1 + nbyte1/2880
         i2 = i+1 + nbyte2/2880
         npt = laxis(1)*laxis(2)
         endif
 
      do i = i1, i2, 1
         read(10,rec=i,iostat=ios) buffc
         nbyte0 = (i-ifirst)*2880+   1
         nbyteE = (i-ifirst)*2880+2880
         np1 = (nbyte0-nbyte1)/4 + 1
         np2 = (nbyteE-nbyte1)/4 + 1
         call buff2pix_r4_wfc(buffb,pix,np1,npt)
         enddo
 
      return
 
 900  continue
      print*,'readfits_WFC: READFITS ERROR'
      print*,'FILE : ',FILE
      print*,'FILEu: ',FILEu
      print*,'  ios: ',ios
      stop
 
      end
 
 
      subroutine buff2pix_r4_wfc(buff,pix,n1,nt)
      implicit none
      byte buff(2880)
      real pix(*)
      integer n1,nt
 
      byte b(4)
      real r
      equivalence(r,b)
 
      integer i, npu, nbu
 
      do i = 1, 720
         npu = n1+i-1
         nbu = (i-1)*4
         if (.not.(_LINUX_)) then
            b(1) = buff(nbu+1)
            b(2) = buff(nbu+2)
            b(3) = buff(nbu+3)
            b(4) = buff(nbu+4)
            endif
         if ((_LINUX_)) then
            b(4) = buff(nbu+1)
            b(3) = buff(nbu+2)
            b(2) = buff(nbu+3)
            b(1) = buff(nbu+4)
            endif
         if (npu.ge.1.and.npu.le.nt) pix(npu) = r
         enddo
 
      return
      end
 
 
 
      subroutine sub_wfcacs_readraw2raz(FILEI,pixz)
      implicit none
 
      character*80 FILEI
      real*4       pixz(8288,2068)
 
      character*80 STRING_HEAD(720)
 
c----------------------
c
c progression of fix...
c
      real, dimension(:,:,:), allocatable ::  pixa0 ! original amp real pixa1(2072,2068,4) ! corrected for ref-pixels
      real, dimension(:,:,:), allocatable ::  pixa1 ! ref pixels corrected for column
      real, dimension(:,:,:), allocatable ::  pixa2 ! whole image corrected for average
      real, dimension(:,:,:), allocatable ::  pixa3 ! whole image corrected for column/average/row
 
      integer i,j,k
 
 
      real*8 barj(2068)
      real*8 sigj(2068)
 
      real    refbar(4), refsig(4)
      integer refuse(4)
 
      real   refcolbar(24,4)
      real   refcolsig(24,4)
 
      real*8  xtot
      real*8  ptot, ptotk
      integer ntot
      real*8  bark(4), barb, bars
      real*8  sigk(4)
      real    dx
 
      real    rj
      integer jj
      real    fj
 
      byte    rowuse(2068,4)
      real    rowref(2068,4)
      real    rowbar(2068,4)
      integer rowtot(4)
 
      byte    usek(2068,4)
 
      integer nskip
      integer nfixd
 
      character*80 FILEB
      character*80 FILEO
 
      integer n
      integer Ls, Lu
      real*8  xl(9999), xbar, xsig
      real*8  bias_shift_bar_kn(4,16)
      real*8  bias_shift_sig_kn(4,16)
      integer iargc, NARG, NARGs
 
      real rn, fn
      integer  nn
 
      real bias_shift(4,17)
      data bias_shift /
     .       0.0000,   0.0000,   0.0000,   0.0000,
     .       2.7270,   2.6762,   2.4282,   2.4256,
     .       5.2498,   5.1106,   4.4137,   4.7800,
     .       7.3278,   7.0069,   5.6505,   7.0362,
     .       9.0420,   8.2736,   6.4285,   8.7642,
     .      10.3455,   9.1745,   7.0294,   9.8053,
     .      11.4173,   9.7687,   7.5445,  10.5576,
     .      12.3301,  10.1248,   8.0041,  11.2053,
     .      13.0142,  10.5457,   8.3000,  11.9011,
     .      13.7405,  10.9688,   8.3529,  12.5295,
     .      14.0843,  11.1583,   8.5229,  13.0714,
     .      14.4022,  11.2714,   8.6627,  13.5581,
     .      15.0245,  11.3666,   8.8438,  14.0528,
     .      15.5655,  11.5520,   9.0489,  14.4749,
     .      15.9700,  11.7052,   9.1862,  14.6787,
     .      16.1995,  11.7212,   9.2854,  14.9770,
     .      16.4175,  11.8703,   9.1763,  15.4222/
 
      real*4, dimension(:,:), allocatable :: pix1
      real*4, dimension(:,:), allocatable :: pix4
 
      integer GAIN
      real    r4_query_hdre
 
 
 
      allocate(pixa0(2072,2068,4))
      allocate(pixa1(2072,2068,4))
      allocate(pixa2(2072,2068,4))
      allocate(pixa3(2072,2068,4))
 
      allocate(pix1(4144,2068))
      allocate(pix4(4144,2068))
 
      if (FILEI(10:18).ne.'_raw.fits') then
          print*,' '
          print*,'FILENAME must have the string "_raw.fits" '
          print*,'         characters 10 through 18.'
          print*,' '
          print*,'         restrictive, I know... '
          print*,' '
          stop
          endif
 
 
c------------------------------------------------------
c
c output image will have _rrs.fits instead of _raw.fits
c
      FILEB = FILEI
      FILEO = FILEI
      FILEO(11:13) = 'raz'
 
      write(*,*)
      write(*,'(''INPUT   INT*2: '',a18)') FILEI(01:18)
      write(*,'(''OUTPUT REAL*4: '',a18)') FILEO(01:18)
      write(*,*)
 
c------------------------------------------------------
c
c read-in the pixel data into 3-d arrays: (2072x2068x4)
c one per amplifier
c
      print*,'read_acsraw_pixa...'
      call read_acsraw_pixa(FILEI,pixa0)
 
      GAIN = 2
      GAIN = r4_query_hdre(FILEI,'CCDGAIN ',-1) + 0.5
 
 
c------------------------------------------------------
c
c STEP1: compute the column-by-column residual for each
c        column in the reference pixels of each amplifier
c        remove the trend so that the ref pixels are flat.
c
      print*,'------------------------------------------------------'
      print*,' '
      print*,'STEP1:  FIND COLUMN BY COLUMN DEPENDENCE...'
      print*,' '
      do k = 1, 4
         call find_refcol(pixa0(1,1,k),refcolbar(1,k),refcolsig(1,k))
         call subt_refcol(pixa0(1,1,k),refcolbar(1,k),
     .                    pixa1(1,1,k))
         enddo
 
 
 
c
c this output just shows the column-by-column trends for
c each amplifier, and also shows how similar the columns
c are with respect to one another.
c
      print*,' '
      print*,'        AMP --- AVERAGE COLUMN RESPONSE (DNs)'
      print*,' '
      write(*,'(9x,1x,3x,24(1x,''i='',i2.2))') (i,i=1,24)
      do k = 1, 4
         write(*,'(9x,i1,3x,24(1x,f4.1))') k,(refcolbar(i,k),i=1,24)
         enddo
 
      print*,' '
      print*,'        AMP --- RMS BY COLUMN (DNs)'
      print*,' '
      write(*,'(9x,1x,3x,24(1x,''i='',i2.2))') (i,i=1,24)
      do k = 1, 4
         write(*,'(9x,i1,3x,24(1x,f4.1))') k,(refcolsig(i,k),i=1,24)
         enddo
      print*,' '
 
 
 
 
 
 
c---------------------------------------------------------
c
c STEP2: Figure out which rows have too much flux to use as
c        reference pixels; at some point if we get the bias
c        drift calibrated, then we may be able to make some
c        of the reference pixels usable again; but until then,
c        it's best to just flag as unusable those that have
c        too much signal; for this, I use an average of 75 DN,
c        which amounts to about 0.4 DN of throw in the bias.
c
      print*,'------------------------------------------------------'
      print*,' '
      print*,'STEP2:  FIND COMPROMISED ROWS...'
      print*,' '
      print*,'        AMP   ROW_USE  ROW_REJ'
      do k = 1, 4
         rowuse(1,k) = 0 ! never use the first row
         rowtot(k) = 0
         ptot = 0.
         do i = 0001, 0024
            ptot = ptot + pixa1(i,1,k)
            enddo
         rowref(1,k) = ptot/24
         rowbar(1,k) = 0.00
         do j = 0002, 2068
            ptot = 0.
            do i = 0001, 0024
               ptot = ptot + pixa1(i,j-1,k)
               ptot = ptot + pixa1(i,j  ,k) ! use avg ref corr...
               enddo
            rowref(j,k) = ptot/48
            ptot = 0.
            do i = 0025, 2072
               ptot = ptot + (pixa1(i,j,k)-rowref(j,k))
               enddo
            rowbar(j,k) = ptot/2048
            rowuse(j,k) = 1
c
c conditions under which we should not use this row to
c tell us about the row-by-row 1/f-noise correction
c
            if (rowbar(j,k).gt.99)      rowuse(j,k) = 0  ! average pixel value of less than 100 e-...
            if (pixa1(25,j,k).gt.25000) rowuse(j,k) = 0  ! near-saturated pixel that may have bled into...
            if (pixa1(26,j,k).gt.25000) rowuse(j,k) = 0
            if (pixa1(27,j,k).gt.25000) rowuse(j,k) = 0
            if (pixa1(28,j,k).gt.25000) rowuse(j,k) = 0
            if (pixa1(29,j,k).gt.25000) rowuse(j,k) = 0
            if (pixa1(30,j,k).gt.25000) rowuse(j,k) = 0
            rowtot(k) = rowtot(k) + rowuse(j,k)
            enddo
         write(*,'(10x,i1,7x,i4,5x,i4)') k,rowtot(k),2067-rowtot(k) ! report the number of good rows per amp
         enddo
      print*,' '
 
 
 
c--------------------------------------------------------
c
c STEP3: Compute the average DC offset for each amplifier
c        Use the good reference rows to determine the
c        average ref-pixel correction for each amplifier.
c        Then remove it.  There is double insurance against
c        bad rows coming in.  We start using only the good
c        rows, but we also find a robust, sigma-clipped
c        mean of the remaining rows.
c
      print*,'------------------------------------------------------'
      print*,' '
      print*,'STEP3:  AVERAGE REFERENCE SUB (DNs)'
      print*,' '
      write(*,119)
  119 format(12x,'AMP',3x,'   BLEV ',7x,'   SIGMA',
     .                 3x,'NREF_USE',3x,'NREF_REJ')
  118 format(12x,i3,   3x,f8.2,      7x,f8.2,
     .                 3x,i8,3x,i8)
      do k = 1, 4
         call find_refbar(pixa1(1,1,k),rowuse(1,k),
     .                    refbar(k),refsig(k),refuse(k),6.0)
         call subt_refbar(pixa1(1,1,k),refbar(k),
     .                    pixa2(1,1,k))
         write(*,118) k,refbar(k),refsig(k),  ! output the RMS
     .                            refuse(k),
     .                    24*2068-refuse(k)
         enddo
      print*,' '
 
 
c----------------------------------------------
c
c this is not needed for the BLEV_COR stage;
c it's a diagnostic to help me see how the bias
c drift correlates with signal in the pixel
c
c     call dump_testf(pixa2,NARG)
 
 
 
c--------------------------------------------------
c
c STEP4: find the line-by-line reference correction;
c        use only the lines that we believe should be
c        good from each amplifier
c
      print*,'------------------------------------------------------'
      print*,' '
      print*,'STEP4:  DEFINE REF CORRECTION LINE BY LINE...'
      print*,'        (assume all 4 amps are the same)     '
 
      nskip = 0
      nfixd = 0
      do j = 0001, 2068 ! go through line by line...
         do k = 01, 04
            ptot = 0.
            do i = 01, 24
               ptot = ptot + pixa2(i,j,k)
               enddo
            bark(k) = ptot/24
            enddo
         do k = 01, 04
            ptot = 0.
            do i = 01, 24
               ptot = ptot + (pixa2(i,j,k)-bark(k))**2
               enddo
            sigk(k) = sqrt(ptot/(24-1))
            enddo
         do k = 1, 4
            usek(j,k) = 0
            if (sigk(k).lt.7.5.and.rowuse(j,k).eq.1) usek(j,k) = 1
            enddo
c
c usek(j,k) now contains 1 for the amplifiers where the row
c is usable; things can be unusable either because
c (1) they have saturated pixels too close
c (2) the flux in that row is high enough to make
c     the bias-drift correction significant
c (3) the rms of the 24 pixels within the ref row are
c     greater than 7.5 DN (typical is 2)
c
         ptot = 0.
         ntot = 0
         do k = 1, 4
            ptot = ptot + usek(j,k)*bark(k)
            ntot = ntot + usek(j,k)
            enddo
         barj(j) = 0.
         sigj(j) = 0.
         if (ntot.ge.2) then                       ! give this row a row-specific
            barj(j) = ptot/ntot                    ! correction only if there are
            ptot = 0.                              ! at least two amps contributing,
            do k = 1, 4                            ! otherwise, use no row correction
               ptot = ptot + usek(j,k)*(bark(k)-barj(j))**2
               enddo
            sigj(j) = sqrt(ptot/(ntot-1))/sqrt(1.*ntot)
            endif
c        write(77,177) NARG,j,barj(j), sigj(j), ntot, ! this is just diagnostic; it allows
c    .                       (bark(k),k=1,4),         ! me to plot the correction implied for
c    .                       (sigk(k),k=1,4),         ! a given row for amp1 against that for
c    .                       (usek(j,k),k=1,4),       ! amp2; they should be well correlated
c    .                       (rowbar(j,k),k=1,4)
c    .
c 177    format(i4,1x,i4.4,1x,f8.4,1x,f8.4,1x,i1,
c    .             5x,4f8.4,
c    .             5x,4f8.4,
c    .             5x,4i2,
c    .             5x,f8.1)
         if (ntot.ge.2) nfixd = nfixd + 1  ! keep a tally of how many rows have the 1/f
         if (ntot.lt.2) nskip = nskip + 1  !    correction performed
         enddo
 
      write(*,*)
      write(*,'(10x,''  ROWS FIXED: '',i4)') nfixd
      write(*,'(10x,''  ROWS SKIPD: '',i4)') nskip
      write(*,*)
 
c
c Now that we have the row-based correction, determine which
c correction to use for each pixel.  Pixels in the data image
c are somewhere between the reference pixels of their row and
c those of the row above; use linear interpolation to determine
c how to weight them.  I could use a spline to do this, but
c maybe that's overkill...
c
      do i = 0001, 2072
      do j = 0001, 2068
         rj = j + (i-12.50)/(2072.0+146.0)
         jj = int(rj)
         if (jj.lt.0001) jj = 0001
         if (jj.gt.2067) jj = 2067
         fj = rj-jj
         barb = barj(jj) + fj*(barj(jj+1)-barj(jj))
         do k =    1,    4
            pixa3(i,j,k) = pixa2(i,j,k) - barb
            enddo
         enddo
         enddo
      print*,' '
 
 
 
c--------------------------------------------------
c
c STEP5: output the results
c
      print*,'------------------------------------------------------'
      print*,'                              '
      print*,'STEP5:  OUTPUT THE RESULTS... '
      print*,'                              '
 
c---------------------------------------------------
c
c The full correction.  Includes
c (1) column correction for each amp,
c (2) average for ref-pixel correction for each amp,
c (3) and the average row-by-row correction over
c     the four amps
c
      call pixa2pix14(pixa3,pix1,pix4)                    ! copies the by-amp cube into by-chip images
 
      do i = 0001, 2072
      do j = 0001, 2068
         pixz(i+(1-1)*2072,j) = pixa3(i,j,1)*2.0170000E+00 ! calibrated gain for amplifier C  ATODGNC
         pixz(i+(2-1)*2072,j) = pixa3(i,j,2)*2.0109999E+00 ! calibrated gain for amplifier D  ATODGND
         pixz(i+(3-1)*2072,j) = pixa3(i,j,3)*2.0200000E+00 ! calibrated gain for amplifier A  ATODGNA
         pixz(i+(4-1)*2072,j) = pixa3(i,j,4)*1.8860000E+00 ! calibrated gain for amplifier B  ATODGNB
         enddo
         enddo
 
      if (GAIN.eq.1) then
          do i = 0001, 8288
          do j = 0001, 2068
             pixz(i,j) = pixz(i,j)/2
             if (pixz(i,j).gt.50000) pixz(i,j) = 65000
             enddo
             enddo
          endif
 
      do k = 1, 4
         do i = 0001, 2048
            rn = 1 + i/128.0
            nn = int(rn)
            fn = rn-nn
            xbar =     bias_shift(k,nn)
     .           + fn*(bias_shift(k,nn+1)
     .                -bias_shift(k,nn  ))
            do j = 0001, 2068
               pixz(24+(k-1)*2072+i,j) =
     .         pixz(24+(k-1)*2072+i,j) - xbar
               enddo
            enddo
         enddo
 
 
      if (.false.) then
         print*,' '
         print*,' '
         do n = 01, 16
            do k = 1, 4
               Ls = 0
               do i = max(0001,(n-2)*128+64),
     .                min(2048,(n-1)*128+64)
               do j = 2063, 2068
                  Ls = Ls + 1
                  xl(Ls) = pixz(24+(k-1)*2072+i,j)
                  enddo
                  enddo
               call barsiga(xl,Ls,xbar,xsig,2.25,Lu)
               bias_shift_bar_kn(k,n) = xbar
               bias_shift_sig_kn(k,n) = xsig
               enddo
            write(91,'(i3,1x,i2,5x,4f10.3,5x,4f10.3,5x,4f10.3)')
     .                NARG,n,
     .            (bias_shift_bar_kn(k,n),k=1,4),
     .               (bias_shift_sig_kn(k,n),k=1,4),
     .               (bias_shift_bar_kn(k,n)
     .               -bias_shift_bar_kn(k,1),k=1,4)
            write(* ,'(i3,1x,i2,5x,4f10.3,5x,4f10.3,5x,4f10.3)')
     .                NARG,n,
     .               (bias_shift_bar_kn(k,n),k=1,4),
     .               (bias_shift_sig_kn(k,n),k=1,4),
     .               (bias_shift_bar_kn(k,n)
     .               -bias_shift_bar_kn(k,1),k=1,4)
            enddo
         print*,' '
         print*,' '
         endif
 
 
c     call cowritfits_raw_r4w(FILEI,FILEO,pix1,pix4)      ! output the chips, using the fits-header of the input image
c     call writfits_r4h(FILEO,pixz,8288,2068,STRING_HEAD)
 
 
c---------------------------------------------------
c
c all but the row-by-row correction...
c
c     call pixa2pix14(pixa2,pix1,pix4)
c     call cowritfits_raw_r4w(FILEI,FILEB,pix1,pix4)
 
      deallocate(pixa0)
      deallocate(pixa1)
      deallocate(pixa2)
      deallocate(pixa3)
 
      return
      end
 
 
 
 
c---------------------------------------------
c
c convert from x-extended format to amp format
c
      subroutine pixb2pixa(pixb0,pixa0)
      implicit none
 
      real pixb0(8288,2068)
      real pixa0(2072,2068,4)
 
      integer i, j, k
 
      do k = 1, 4
         do i = 0001, 2072
         do j = 0001, 2068
            pixa0(i,j,k) = pixb0(i+(k-1)*2072,j)
            enddo
            enddo
         enddo
 
      return
      end
 
 
c---------------------------------------------
c
c convert from amp format to x-extended format
c
      subroutine pixa2pixb(pixa,pixb)
      implicit none
 
      real pixa(2072,2068,4)
      real pixb(8288,2068)
 
      integer i, j, k
      do k = 1, 4
         do i = 0001, 2072
         do j = 0001, 2068
            pixb(i+(k-1)*2072,j) = pixa(i,j,k)
            enddo
            enddo
         enddo
 
      return
      end
 
 
 
 
 
 
 
 
 
c--------------------------------------------------------------
c
c this will read-in a standard-format _raw image into a
c three-dimensional array, 2072x2068 for each of the 4 amps
c
      subroutine read_acsraw_pixa(FILENAME,pix_ijk)
      implicit none
 
      character*80 FILENAME
      real pix_ijk(2072,2068,4)
 
      integer*2, dimension(:,:), allocatable :: pixi
 
      integer i, j
 
      print*,'enter read-acsraw_pixa...'
      allocate(pixi(4144,2068))
 
      call readfits_i2e(FILENAME,pixi,4144,2068,1)
 
      do i = 0001, 2072
      do j = 0001, 2068
         pix_ijk(i,j,1) = pixi(0000+i,0000+j)+32768
         pix_ijk(i,j,2) = pixi(4145-i,0000+j)+32768
         enddo
         enddo
 
      call readfits_i2e(FILENAME,pixi,4144,2068,4)
 
      do i = 0001, 2072
      do j = 0001, 2068
         pix_ijk(i,j,3) = pixi(0000+i,2069-j)+32768
         pix_ijk(i,j,4) = pixi(4145-i,2069-j)+32768
         enddo
         enddo
 
      deallocate(pixi)
 
      return
      end
 
 
c--------------------------------------------------------------
c
c this routine will subtract the average reference-pixel value
c from each amplifer
c
      subroutine subt_refbar(pix_ij_bef,refbar,
     .                       pix_ij_aft)
      real pix_ij_bef(2072,2068)
      real refbar
      real pix_ij_aft(2072,2068)
 
      integer i, j
 
      do j = 0001, 2068
      do i = 0001, 2072
         pix_ij_aft(i,j) = pix_ij_bef(i,j) - refbar
         enddo
         enddo
 
      return
      end
 
 
 
c--------------------------------------------------------------
c
c
c
      subroutine avg_overk(psplink,psplinb,psplins,Ns)
      implicit none
 
      integer Ns
      real psplink(Ns,4)
      real psplinb(Ns)
      real psplins(Ns)
 
      integer n
 
      do n = 1, Ns
         psplinb(n) = (psplink(n,1)+psplink(n,2)
     .                +psplink(n,3)+psplink(n,4))/4.0
         psplins(n) = (abs(psplink(n,1)-psplinb(n))
     .                +abs(psplink(n,2)-psplinb(n))
     .                +abs(psplink(n,3)-psplinb(n))
     .                +abs(psplink(n,4)-psplinb(n)))/3.0
         enddo
 
      return
      end
 
 
c--------------------------------------------------------------
c
c
      subroutine subt_refspline(pix_ij,vsplinb)
      implicit none
 
      real pix_ij(2072,2068)
      real vsplinb(2068)
 
      integer i, j
      real vcor
      real f
 
      do j = 0001, 2068
      do i = 0001, 2072
         f = (i-12.5)/(2072.00-12.5)
         if (f.lt.0) f = 0.0
         vcor = vsplinb(2068)
         if (i.le.2067) then
            vcor = (1-f)*vsplinb(j  )
     .           + ( f )*vsplinb(j+1)
            endif
         pix_ij(i,j) = pix_ij(i,j) - vcor
         enddo
         enddo
 
      return
      end
 
 
 
 
      subroutine pixa2pix14(pix_ijk,pix1,pix4)
      implicit none
 
      real pix_ijk(2072,2068,4)
      real pix1(4144,2068)
      real pix4(4144,2068)
 
      integer i, j
 
      do i = 0001, 2072
      do j = 0001, 2068
         pix1(0000+i,0000+j) = pix_ijk(i,j,1)
         pix1(4145-i,0000+j) = pix_ijk(i,j,2)
         enddo
         enddo
 
      do i = 0001, 2072
      do j = 0001, 2068
         pix4(0000+i,2069-j) = pix_ijk(i,j,3)
         pix4(4145-i,2069-j) = pix_ijk(i,j,4)
         enddo
         enddo
 
      return
      end
 
 
c-------------------------------------------------------------
c
c
      subroutine copy_pixa(pixa1,pixa2)
      implicit none
 
      real pixa1(2072,2068,4)
      real pixa2(2072,2068,4)
 
      integer i, j, k
 
      do k = 1, 4
         do i = 0001, 2072
         do j = 0001, 2068
            pixa2(i,j,k) = pixa1(i,j,k)
            enddo
            enddo
         enddo
 
      return
      end
 
 
 
 
c-----------------------------------------------------
c
c will write a real*4 into the header-shell of an
c integer*2 raw image (FILEI)
c
c There are 6 extensions total, but only #1 and #4
c have any data.
c
      subroutine cowritfits_raw_r4w(FILEI,FILEO,pix1,pix4)
      implicit none
 
      character*80 FILEI, FILEO
      real*4 pix1(4096,2048)
      real*4 pix4(4096,2048)
 
      integer nimg
      integer naxes
      integer laxis(3)
 
      character*8  field
      character*20 stream
 
      integer nbyte0
      integer nbyteE
      integer nbperi, nbyte1i, nbyte2i, npti
      integer nbperj, nbyte1j, nbyte2j, nptj
      integer ios, k, kk
 
      character*2880 buffc
      byte buffb(2880)
      equivalence (buffc,buffb)
 
 
      integer np1, np2
      integer NREAD
      integer bitpix
 
c     real dtde_l(11)
c     real chg_leak(4,17)
c     common /dtde_l_  /dtde_l
c     common /chg_leak_/chg_leak
 
      logical atend
 
      real rDAT
      character*12 DATESTR
      character*12 TIMESTR
      common /rDAT_/rDAT,DATESTR,TIMESTR
 
      real CTE_FRAC
      common /CTE_FRAC_/CTE_FRAC
 
      integer NMOD
      common /MMOD_/NMOD
      character*11 HOW(4)
 
      integer NIT2DO
      common/NIT2DO_/NIT2DO
 
      real pix1min, pix1max
      real pix4min, pix4max
      integer i, ifirst, i1, i2
      integer j, jfirst, j1, j2
 
      logical DIAG
      common /DIAG_/DIAG
      data DIAG/.false./
 
 
      pix1min = 0.0
      pix1max = 0.0
      pix4min = 0.0
      pix4max = 0.0
      do i = 0001, 4144
      do j = 0001, 2068
         if (pix1(i,j).lt.pix1min) pix1min = pix1(i,j)
         if (pix4(i,j).lt.pix4min) pix4min = pix4(i,j)
         if (pix1(i,j).gt.pix1max) pix1max = pix1(i,j)
         if (pix4(i,j).gt.pix4max) pix4max = pix4(i,j)
         enddo
         enddo
 
      if (DIAG) then
         print*,'---> pix1min: ',pix1min, pix1max
         print*,'---> pix4min: ',pix4min, pix4max
         endif
 
      HOW(1) = 'NONE       '
      HOW(2) = 'LINEAR-3   '
      HOW(3) = 'QUADRATIC-5'
      HOW(4) = 'LINEAR-5   '
 
      if (.true.) then
         print*,'        ENTER COWRITE_RAW: '
         print*,'              ----> FILEI: ',FILEI(1:30)
         print*,'              ----> FILEO: ',FILEO(1:30)
         print*,'                           '
         endif
 
      if (DIAG) print*,'OPEN10 FILEI: ',FILEI
      open(10,file=FILEI,status='old',
     .     err=900,recl=2880,form='UNFORMATTED',
     .     access='DIRECT')
 
      HOW(1) = 'NONE       '
      HOW(2) = 'LINEAR-3   '
      HOW(3) = 'QUADRATIC-5'
      HOW(4) = 'LINEAR-5   '
 
      if (DIAG) then
         print*,'ENTER COWRITE_RAW: '
         print*,'      ----> FILEI: ',FILEI(1:30)
         print*,'      ----> FILEO: ',FILEO(1:30)
         endif
 
      if (DIAG) print*,'OPEN10 FILEI: ',FILEI
      open(10,file=FILEI,status='old',
     .     err=900,recl=2880,form='UNFORMATTED',
     .     access='DIRECT')
 
      if (DIAG) print*,'OPEN11 FILEO: ',FILEO
      open(11,file=FILEO,status='unknown',
     .     err=900,recl=2880,form='UNFORMATTED',
     .     access='DIRECT')
 
      i = 0
      j = 0
      NREAD = -1
 099  continue
      NREAD = NREAD + 1
      if (DIAG) print*,'   '
      naxes =  0
      laxis(1) = 1
      laxis(2) = 1
      laxis(3) = 1
 100  continue
      i = i + 1
      j = j + 1
      if (DIAG) write(*,'(80(''-''))')
      read(10,rec=i,iostat=ios) buffc
      atend = .false.
      do k = 00, 35
         if (DIAG)
     .   write(*,'(''NREAD='',i1,3x,
     .             ''IREC='',i9.9,1x,
     .             ''JREC='',i9.9,1x,i2,1x,a80)')
     .               NREAD,i,j,k,buffc(k*80+1:k*80+80)
         field  = buffc(k*80+01:k*80+08)
         stream = buffc(k*80+11:k*80+31)
 
 
 
         if (NREAD.eq.1.or.NREAD.eq.4) then
            if (field.eq.'BITPIX  ') write(buffc(k*80+01:k*80+80),111)
  111             format('BITPIX  =                  -32 / bits pe',
     .                   'r data value                            ')
            if (field.eq.'BZERO   ') write(buffc(k*80+01:k*80+80),112)
  112             format('BZERO   =                  0.0 / physica',
     .                   'l value for an array value of zero      ')
            endif
         if (NREAD.eq.1) then
            if (field.eq.'DATAMIN ')
     .          write(buffc(k*80+01:k*80+80),113) pix1min
            if (field.eq.'DATAMAX ')
     .          write(buffc(k*80+01:k*80+80),114) pix1max
            endif
         if (NREAD.eq.4) then
            if (field.eq.'DATAMIN ')
     .          write(buffc(k*80+01:k*80+80),113) pix4min
            if (field.eq.'DATAMAX ')
     .          write(buffc(k*80+01:k*80+80),114) pix4max
            endif
  113    format('DATAMIN =             ',f8.2,' / physica',
     .          'l value for an array value of zero      ')
  114    format('DATAMAX =             ',f8.2,' / physica',
     .          'l value for an array value of zero      ')
         if (field.eq.'NAXIS   ') read(stream,*) naxes
         if (field.eq.'NAXIS1  ') read(stream,*) laxis(1)
         if (field.eq.'NAXIS2  ') read(stream,*) laxis(2)
         if (buffc(k*80+01:k*80+3).eq.'END') atend = .true.
         enddo
      if (.not.atend) then
         write(11,rec=j,iostat=ios) buffc
         goto 100
         endif
      write(11,rec=j,iostat=ios) buffc
 
      if (DIAG) then
         print*,' '
         print*,'PROCESS...'
         print*,'   NREAD = ',NREAD
         print*,'   NAXIS = ',NAXES
         print*,'   LAXIS1= ',laxis(1)
         print*,'   LAXIS2= ',laxis(2)
         print*,' '
         endif
 
      if (NREAD.eq.6) then
         close(10)
         close(11)
         return
         endif
 
      if (naxes.eq.0) goto 099
 
      ifirst = i+1 ! first record of the image in this extension
      jfirst = j+1 ! first record of the image in this extension
 
      nbperi  = 2*laxis(1)*laxis(2)
      npti    =   laxis(1)*laxis(2)
      nbyte1i = 1
      nbyte2i = nbperi
      i1 = i+1 + nbyte1i/2880
      i2 = i+1 + nbyte2i/2880
 
      nbperj  = 4*laxis(1)*laxis(2)
      nptj    =   laxis(1)*laxis(2)
      nbyte1j = 1
      nbyte2j = nbperj
      j1 = j+1 + nbyte1j/2880
      j2 = j+1 + nbyte2j/2880
 
      if (NREAD.eq.2.or.NREAD.eq.3.or.
     .    NREAD.eq.5.or.NREAD.eq.6) then
          print*,'NREAD = ',NREAD
          print*,'naxes = ',naxes
          print*,'we shouldnt be here for anything'
          print*,'but NREAD=1 and NREAD=4 '
          stop
          endif
 
      if (NREAD.eq.1) then
         if (DIAG) print*,'NREAD = 1... OUTPUT IMAGE'
         do j = j1, j2, 1
            nbyte0 = (j-jfirst)*2880+   1
            nbyteE = (j-jfirst)*2880+2880
            np1 = (nbyte0-nbyte1j)/4 + 1
            np2 = (nbyteE-nbyte1j)/4 + 1
            call pix2buff_r4_(buffb,pix1,np1,nptj)
            write(11,rec=j,iostat=ios) buffc
            enddo
         i = i2
         j = j2
         goto 099
         endif
 
      if (NREAD.eq.4) then
         if (DIAG) print*,'NREAD = 4... OUTPUT IMAGE'
         do j = j1, j2, 1
            nbyte0 = (j-jfirst)*2880+   1
            nbyteE = (j-jfirst)*2880+2880
            np1 = (nbyte0-nbyte1j)/4 + 1
            np2 = (nbyteE-nbyte1j)/4 + 1
            call pix2buff_r4_(buffb,pix4,np1,nptj)
            write(11,rec=j,iostat=ios) buffc
            enddo
         i = i2
         j = j2
         goto 099
         endif
 
      print*,'cowrite_raw_rbc should never get here...'
      stop
 
      return
 
 900  continue
      print*,'cowritefits_raw: READFITS ERROR'
      stop
 
      end
 
 
 
c-------------------------------------------------------
c
c
      subroutine pix2buff_r4_(buff,pix,n1,nt)
      implicit none
      byte buff(2880)
      real*4 pix(*)
      integer n1,nt
 
      byte b(4)
      real*4 r
      equivalence(r,b)
 
      integer i, npu, nbu
 
      do i = 1, 720
         npu = n1+i-1
         nbu = (i-1)*4
         if (npu.ge.1.and.npu.le.nt) r = pix(npu)
         if (.not.(_LINUX_)) then
            buff(nbu+1) = b(1)
            buff(nbu+2) = b(2)
            buff(nbu+3) = b(3)
            buff(nbu+4) = b(4)
            endif
         if ((_LINUX_)) then
            buff(nbu+1) = b(4)
            buff(nbu+2) = b(3)
            buff(nbu+3) = b(2)
            buff(nbu+4) = b(1)
            endif
         enddo
 
      return
      end
 
 
 
c------------------------------------------------
c
c find the average value of the reference pixels;
c with sigma clipping at SIG_CLIP
c
c
      subroutine find_refbar(pix_ij,rowuse,
     .                       refbar,refsig,refuse,SIG_CLIP)
      implicit none
 
      real    pix_ij(2072,2068)
      byte    rowuse(2068)
      real    refbar
      real    refsig
      integer refuse
      real SIG_CLIP
 
 
      integer i, j
      real*8  stot
      real*8  ptot
      integer ntot
      integer ntoto
      real*8  bar, sig
      byte    pixu(24,2068)
      integer NIT
 
      ntot = 0
      do i = 0001, 0024
      do j = 0001, 2068
         pixu(i,j) = rowuse(j)
         ntot = ntot + pixu(i,j)
         enddo
         enddo
 
      NIT = 0
    1 continue
      ntoto = ntot
      NIT = NIT + 1
 
      ptot = 0.
      ntot = 0
      do i = 0001, 0024
      do j = 0001, 2068
         ptot = ptot + pixu(i,j)*pix_ij(i,j)
         ntot = ntot + pixu(i,j)
         enddo
         enddo
      bar = ptot/ntot
 
      stot = 0.
      do i = 0001, 0024
      do j = 0001, 2068
         stot = stot + pixu(i,j)*abs(pix_ij(i,j)-bar)
         enddo
         enddo
      sig = stot/ntot
 
      ntot = 0
      do i = 0001, 0024
      do j = 0001, 2068
         if (abs(pix_ij(i,j)-bar).gt.SIG_CLIP*sig) pixu(i,j) = 0
         ntot = ntot + pixu(i,j)
         enddo
         enddo
      close(31)
 
      if (NIT.le.05) goto 1
      if (NIT.le.10.and.ntot.lt.ntoto) goto 1  ! iterate max 10 times; only repeat if we've rejected pixels
 
      refbar = bar
      refsig = sig
      refuse = ntot
 
 
      stot = 0.
      ntot = 0
      do i = 0001, 0024
      do j = 0001, 2068
         stot = stot + pixu(i,j)*(pix_ij(i,j)-bar)**2
         ntot = ntot + pixu(i,j)
         enddo
         enddo
      refsig = sqrt(stot/ntot)
 
      return
      end
 
 
 
 
c--------------------------------------------------------------
c
c this will find the column-by-column residual for each row
c      refcolbar(i) is the average of each column relative
c      refcolsig(i) is the sigma along the column
c
      subroutine find_refcol(pix_ij,refcolbar,refcolsig)
      implicit none
 
      real pix_ij(2072,2068)
      real refcolbar(24)
      real refcolsig(24)
 
      integer i, j
      real*8  ptot
      real*8  stot
      integer ntot
      real*8  pnorm(24,2068)
      real*8  collist(2068)
      real*8  bar, sig
      integer Ns, Nu
 
c---------------------------------------------
c
c Subtract the average for each row, so that we can
c focus on the column-by-column behavior.  Since
c most of the columns start high and asymtote after
c 12 pixels or so, let's define the average by the
c right half.
c
      do j = 0001, 2068
         ptot = 0.
         do i = 0013, 0024
            ptot = ptot + pix_ij(i,j)
            enddo
         bar = ptot/12
         do i = 0001, 0024
            pnorm(i,j) = pix_ij(i,j) - bar  ! pnorm is the residual
            enddo
         enddo
 
      do i = 01, 24
         Ns = 0
         do j = 0001, 2068
            if (pix_ij(25,j).lt.25000.and.   ! I have found that if there are saturated
     .          pix_ij(26,j).lt.25000.and.   !    pixels near the reference pixels,
     .          pix_ij(27,j).lt.25000.and.   !    they can bleed into the ref pixels
     .          pix_ij(28,j).lt.25000.and.   !    and mess up the column by column
     .          pix_ij(29,j).lt.25000.and.   !    trends ; so only use columns that
     .          pix_ij(30,j).lt.25000) then  !    are free of trends.
                Ns = Ns + 1
                collist(Ns) = pnorm(i,j)
                endif
            enddo
         call barsig_r4(collist,Ns,bar,sig,5.0,Nu) ! this takes the list of good residuals
  199    format(i2,1x,f10.2,1x,f10.3,1x,i4)     !   for column#i and determines the
         refcolbar(i) = bar                     !   average excess/deficit and its rms
         refcolsig(i) = sig                     !   the average will be used to adjust the
         enddo                                  !   column by column ref-pix trends
 
      return
      end
 
 
 
 
c-------------------------------------------------------
c
c
c
      subroutine barsig_r4(xlist,NTOT,bar,sig,SIG_REJ,NUSE)
      implicit none
 
      integer NTOT
      real*8 xlist(NTOT)
      real*8 bar
      real*8 sig
      real   SIG_REJ
      integer NUSE
 
      integer n
      real*8    bsum, ssum
      integer nsum
      integer NIT
 
 
      bar = 0.e0
      sig = 9e9
      do NIT = 1, 5
         bsum = 0.
         ssum = 0.
         nsum = 0.
         do n = 1, NTOT
            if (abs(xlist(n)-bar).le.SIG_REJ*sig) then
               bsum = bsum + xlist(n)
               ssum = ssum + (xlist(n)-bar)**2
               nsum = nsum + 1
               endif
            enddo
         if (nsum.gt.0) bar = bsum / nsum
         if (nsum.gt.1) sig = sqrt(ssum/(nsum-1))
         enddo
      NUSE = nsum
      if (nsum.le.1) sig = 0.999
 
      return
      end
 
 
c-------------------------------------------------------
c
c
c
      subroutine barsiga(xlist,NTOT,bar,sig,SIG_REJ,NUSE)
      implicit none
 
      integer NTOT
      real*8 xlist(NTOT)
      real*8 bar
      real*8 sig
      real   SIG_REJ
      integer NUSE
 
      integer n
      real*8    bsum, ssum
      integer nsum
      integer NIT
 
 
      bar = 0.e0
      sig = 9e9
      do NIT = 1, 15
         bsum = 0.
         ssum = 0.
         nsum = 0.
         do n = 1, NTOT
            if (abs(xlist(n)-bar).le.SIG_REJ*sig) then
               bsum = bsum + xlist(n)
               ssum = ssum + abs(xlist(n)-bar)
               nsum = nsum + 1
               endif
            enddo
         if (nsum.gt.0) bar = bsum / nsum
         if (nsum.gt.1) sig = ssum / (nsum-1)
         enddo
      NUSE = nsum
      if (nsum.le.1) sig = 0.999
 
      return
      end
 
 
c--------------------------------------------------------------
c
c this will subtract the column-by-column trends from the
c reference pixels.
c
      subroutine subt_refcol(pix_ij_BEF,refcolbar,
     .                       pix_ij_AFT)
      implicit none
 
      real pix_ij_BEF(2072,2068)
      real refcolbar(24)
      real pix_ij_AFT(2072,2068)
 
      integer i, j
 
      do i = 0001, 0024
      do j = 0001, 2068
         pix_ij_AFT(i,j) = pix_ij_BEF(i,j) - refcolbar(i)
         enddo
         enddo
 
      do i = 0025, 2072
      do j = 0001, 2068
         pix_ij_AFT(i,j) = pix_ij_BEF(i,j)
         enddo
         enddo
 
      return
      end
 
 
 
c--------------------------------------------------------------
c
c this will output the data needed to optimize the
c correction for bias-drift ; Matt, there's no need
c to translate this routine
c
      subroutine dump_testf(pix_ijk,NARG)
      implicit none
 
      real    pix_ijk(2072,2068,4)
      integer NARG
 
      integer i, j, k, n
 
      real*8  a(20)
      real    f(20)
      real    val
      real*8  ptot, refbar
 
      data f / 0.00100, 0.00095, 0.00090, 0.00085, 0.00080,
     .         0.00075, 0.00070, 0.00065, 0.00060, 0.00055,
     .         0.00050, 0.00045, 0.00040, 0.00035, 0.00030,
     .         0.00025, 0.00020, 0.00015, 0.00010, 0.00005/
 
      do k = 1, 4
         do n = 1, 20
            a(n) = 0.00
            enddo
         do j = 0001, 2068
            ptot = 0
            do i = 01, 24
               ptot = ptot + pix_ijk(i,j,k)
               enddo
            refbar = ptot/24
            write(99,199) NARG,k,j,refbar,(a(n),n=1,20)
  199       format(i3.3,1x,i1,1x,i4.4,3x,f8.2,3x,20(f8.1,1x))
            do i = 0001, 2068+146
               val = 0.00
               if (i.ge.0001.and.i.le.2068) val = pix_ijk(i,j,k)
               do n = 1, 20
                  a(n) = a(n) + (val-a(n))*f(n)
                  enddo
               enddo
            enddo!j
         enddo!k
 
 
      return
      end
 
 
      subroutine readfits_acsraw2r(FILENAME,pix)
      implicit none
 
      character*80 FILENAME
      real         pix(4096,4096)
 
      real         pixz(8288,2068)
 
      integer i, j, k
 
      call sub_wfcacs_readraw2raz(FILENAME,pixz)
 
      do i = 0001, 2048
      do j = 0001, 2048
         pix(i+0000,j+0000) = pixz(0*2072+24+i,j)
         pix(4097-i,j+0000) = pixz(1*2072+24+i,j)
         pix(i+0000,4097-j) = pixz(2*2072+24+i,j)
         pix(4097-i,4097-j) = pixz(3*2072+24+i,j)
         enddo
         enddo
 
      return
      end
 
 
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/FITSIO/ACSHRC/read_acshrc_flt_full.f"
c****                                         
c*********************************************
                                              
 
      subroutine read_acshrc_flt_smpl(FILENAME,pix)
      implicit none
 
      character*80 FILENAME
      real      pix(1024,1024)
 
      real*8    CRPIX1, CRPIX2
      real*8    CRVAL1, CRVAL2
      real*8    CD1_1,  CD1_2
      real*8    CD2_1,  CD2_2
 
      integer   LOFLAG, HIFLAG
      real*8    BDRY_XR(4,4)
      real*8    BDRY_YR(4,4)
 
      character*05 FILT
      real*4       EXPT
      real*4       RDAT
      integer      PROP
 
      call read_acshrc_flt_full(FILENAME,pix,
     .                          CRPIX1, CRPIX2,
     .                          CRVAL1, CRVAL2,
     .                          CD1_1,  CD1_2,
     .                          CD2_1,  CD2_2,
     .                          LOFLAG, HIFLAG,
     .                          BDRY_XR,  BDRY_YR,
     .                          FILT,EXPT,
     .                          RDAT,PROP)
 
      return
      end
 
 
 
      subroutine read_acshrc_flt_full(FILENAME,pix,
     .                                CRPIX1, CRPIX2,
     .                                CRVAL1, CRVAL2,
     .                                CD1_1,  CD1_2,
     .                                CD2_1,  CD2_2,
     .                                LOFLAG, HIFLAG,
     .                                BDRY_XR,  BDRY_YR,
     .                                FILT,EXPT,
     .                                RDAT,PROP)
      implicit none
 
 
 
 
      character*80 FILENAME
      real      pix(1024,1024)
 
      real*8    CRPIX1, CRPIX2
      real*8    CRVAL1, CRVAL2
      real*8    CD1_1,  CD1_2
      real*8    CD2_1,  CD2_2
 
      integer   LOFLAG, HIFLAG
      real*8    BDRY_XR(4,4)
      real*8    BDRY_YR(4,4)
 
      character*05 FILT
      real*4       EXPT
      real*4       RDAT
      integer      PROP
 
      integer NAXIS1
      integer NAXIS2
 
      real*8 COSPA, SINPA
      real*8 PA_APER
      real*8 PA_V3
 
      character*80 FILTU
      character*80 FILT1, FILT2, FILT3, FILT4
      real*8  r8_query_hdre
      real*4  r4_query_hdre
      integer i4_query_hdre
      real*8  rdate_header
 
      integer i, j
 
      print*,'   '
      print*,'ENTER: subroutine read_wfcacs_flt_full '
      print*,'---> FILENAME: ',FILENAME
      print*,'   '
 
      NAXIS1 = r8_query_hdre(FILENAME,'NAXIS1  ',-1)
      NAXIS2 = r8_query_hdre(FILENAME,'NAXIS2  ',-1)
 
      if (NAXIS1.ne.1024.and.NAXIS2.ne.1024)
     .    stop 'only FLT ACS/HRC full frame (for now)'
 
      CRPIX1 = r8_query_hdre(FILENAME,'CRPIX1  ',-1)
      CRPIX2 = r8_query_hdre(FILENAME,'CRPIX2  ',-1)
      CRVAL1 = r8_query_hdre(FILENAME,'CRVAL1  ',-1)
      CRVAL2 = r8_query_hdre(FILENAME,'CRVAL2  ',-1)
 
      PA_APER = r8_query_hdre(FILENAME,'PA_APER ',-1)
      PA_V3   = PA_APER - 0.001
 
      COSPA = cos(PA_V3*3.14159/180.0)
      SINPA = sin(PA_V3*3.14159/180.0)
 
      CD1_1 = -COSPA*0.02829/60/60
      CD1_2 =  SINPA*0.02829/60/60
      CD2_1 =  SINPA*0.02829/60/60
      CD2_2 =  COSPA*0.02829/60/60
 
      PROP = i4_query_hdre(FILENAME,'PROPOSID',-1)
      EXPT = r4_query_hdre(FILENAME,'EXPTIME ',-1)
      RDAT = rdate_header(FILENAME)
 
      call query_hdre(FILENAME,'FILTER  ',FILTU,-1)
      call query_hdre(FILENAME,'FILTER1 ',FILT1,-1)
      call query_hdre(FILENAME,'FILTER2 ',FILT2,-1)
      call query_hdre(FILENAME,'FILTNAM1',FILT3,-1)
      call query_hdre(FILENAME,'FILTNAM2',FILT4,-1)
 
      FILT = FILTU
      if (FILT1(1:1).eq.'F')  FILT = FILT1(1:5)
      if (FILT2(1:1).eq.'F')  FILT = FILT2(1:5)
      if (FILT3(1:1).eq.'F')  FILT = FILT3(1:5)
      if (FILT4(1:1).eq.'F')  FILT = FILT4(1:5)
      if (FILT1(2:2).eq.'F')  FILT = FILT1(2:6)
      if (FILT2(2:2).eq.'F')  FILT = FILT2(2:6)
      if (FILT3(2:2).eq.'F')  FILT = FILT3(2:6)
      if (FILT4(2:2).eq.'F')  FILT = FILT4(2:6)
 
      LOFLAG =    -50
      HIFLAG = 125000
 
      do i = 1, 4
      do j = 1, 4
         BDRY_XR(i,j) = 0.
         BDRY_YR(i,j) = 0.
         enddo
         enddo
 
      BDRY_XR(1,1) = 0002
      BDRY_YR(1,1) = 0002
      BDRY_XR(2,1) = 1023
      BDRY_YR(2,1) = 0002
      BDRY_XR(3,1) = 1023
      BDRY_YR(3,1) = 1023
      BDRY_XR(4,1) = 0002
      BDRY_YR(4,1) = 1023
 
 
      call readfits_HRC(FILENAME,pix,1)
 
      call HRCIMGFIX(pix)
 
      return
      end
 
 
 
 
 
      subroutine readfits_HRC(FILE,pix,nimg)
      implicit none
 
      character*(*) FILE
      real pix(1024,1024)
 
      character*70 INFO(10)
      common / fitsinfo / INFO
 
      integer nimg
      integer naxes
      integer laxis(3)
 
      character*8  field
      character*20 stream
 
      integer nbyte0
      integer nbyteE
      integer nbyte1
      integer nbyte2
      integer nbper
      integer i,ios, k
 
      character*2880 buffc
      byte buffb(2880)
      equivalence (buffc,buffb)
 
      integer ifirst, i1, i2
 
      integer np1, np2, npt
      integer nextend
      integer nread
      real bscale, bzero
      integer bitpix
 
 
      logical LINUX
      data LINUX/.true./
 
      logical first_header
 
      integer NH_RECs
      character*2880 HEADER_STRING(99)
      common /HEADER_STRING_INFO/NH_RECs,HEADER_STRING
 
      open(10,file=FILE,status='old',
     .     err=900,recl=2880,form='UNFORMATTED',
     .     access='DIRECT')
 
      bscale = 1.0
      bzero  = 0.0
 
      naxes = -1
      laxis(1) = 1
      laxis(2) = 1
      laxis(3) = 1
      nextend = 0
 
 
      NH_RECs = 0
      first_header = .true.
      i = 0
      nread = 0
 100  continue
      i = i + 1
      read(10,rec=i,iostat=ios) buffc
      if (first_header) then
         NH_RECs = NH_RECs + 1
         HEADER_STRING(NH_RECs) = buffc
         endif
      do k = 0, 35, 1
         field  = buffc(k*80+01:k*80+08)
         stream = buffc(k*80+11:k*80+31)
         if (field.eq.'NAXIS   ') read(stream,*) naxes
         if (field.eq.'NAXIS1  ') read(stream,*) laxis(1)
         if (field.eq.'NAXIS2  ') read(stream,*) laxis(2)
         if (field.eq.'NAXIS3  ') read(stream,*) laxis(3)
         if (field.eq.'NEXTEND ') read(stream,*) nextend
         if (field.eq.'BITPIX  ') read(stream,*) bitpix
         if (field.eq.'BSCALE  ') read(stream,*) bscale
         if (field.eq.'BZERO   ') read(stream,*) bzero
 
         if (field.eq.'END     ') then
            first_header = .false.
            goto 101
            endif
         enddo
      goto 100
 101  continue
 
 
      nread = nread + 1
 
      ifirst = i+1
      i1 = i
      i2 = i
 
      if (naxes.eq.0) then    ! maybe multiple images stored as extensions
         if (nextend.eq.0) then
            print*,'THIS IS A NULL IMAGE: '
            print*,'NAXES: ',NAXES
            print*,'NEXND: ',NEXTEND
            stop
            endif
         endif
 
      if (nread.ne.nimg+1) then
         nbper  = abs(BITPIX/8)*laxis(1)*laxis(2)
         if (NAXES.eq.0) nbper = 0
         i = i + 1.0*nbper/2880 + 0.9999
         goto 100
         endif
 
      if (laxis(1).ne.1024.or.
     .    laxis(2).ne.1024) then
          print*,'  laxis1: ',laxis(1)
          print*,'  laxis2: ',laxis(2)
          print*,'    1024: ',1024
          print*,'    1024: ',1024
          stop
          endif
 
      if (naxes.eq.2) then   ! nimg is irrelevant; ignore
         nbper  = 4*laxis(1)*laxis(2)
         npt    = laxis(1)*laxis(2)
         nbyte1 = 1
         nbyte2 = nbper
         i1 = i+1 + nbyte1/2880
         i2 = i+1 + nbyte2/2880
         endif
 
      if (naxes.eq.3) then
         nbper  = 4*laxis(1)*laxis(2)
         nbyte1 = 1 + nbper*(nimg-1)
         nbyte2 =     nbper*(nimg  )
         i1 = i+1 + nbyte1/2880
         i2 = i+1 + nbyte2/2880
         npt = laxis(1)*laxis(2)
         endif
 
      do i = i1, i2, 1
         read(10,rec=i,iostat=ios) buffc
         nbyte0 = (i-ifirst)*2880+   1
         nbyteE = (i-ifirst)*2880+2880
         np1 = (nbyte0-nbyte1)/4 + 1
         np2 = (nbyteE-nbyte1)/4 + 1
         call buff2pix_r4_hrc(buffb,pix,np1,npt)
         enddo
 
      return
 
 900  continue
      print*,'readfits_WFC: READFITS ERROR'
      stop
 
      end
 
 
      subroutine buff2pix_r4_hrc(buff,pix,n1,nt)
      implicit none
      byte buff(2880)
      real pix(*)
      integer n1,nt
 
      byte b(4)
      real r
      equivalence(r,b)
 
      integer i, npu, nbu
 
      do i = 1, 720
         npu = n1+i-1
         nbu = (i-1)*4
         if (.not.(_LINUX_)) then
            b(1) = buff(nbu+1)
            b(2) = buff(nbu+2)
            b(3) = buff(nbu+3)
            b(4) = buff(nbu+4)
            endif
         if ((_LINUX_)) then
            b(4) = buff(nbu+1)
            b(3) = buff(nbu+2)
            b(2) = buff(nbu+3)
            b(1) = buff(nbu+4)
            endif
         if (npu.ge.1.and.npu.le.nt) pix(npu) = r
         enddo
 
      return
      end
 
 
c-----------------------------------------------------
c
c black-out the occulting finger, so no stars will be
c sought there
c
      subroutine HRCIMGFIX(pix)
      implicit none
      real pix(1024,1024)
 
      integer i, j, ic
 
      do j = 1024, 0777, -1
         ic = 349 + (432-349)*(1024-j)/(1024-0777)
         do i = ic-15, ic+15
            pix(i,j) = -100
            enddo
         enddo
 
      return
      end
 
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/FITSIO/WFC3UV/read_wfc3uv_flt_full.f"
c****                                         
c*********************************************
                                              
 
 
      subroutine read_wfc3uv_flt_smpl(FILENAME,pix)
      character*80 FILENAME
      real      pix(4096,4096)
 
      real*8    CRPIX1, CRPIX2
      real*8    CRVAL1, CRVAL2
      real*8    CD1_1,  CD1_2
      real*8    CD2_1,  CD2_2
 
      integer   LOFLAG, HIFLAG
      real*8    BDRY_XR(4,4)
      real*8    BDRY_YR(4,4)
 
      character*05 FILT
      real*4       EXPT
      real*4       RDAT
      integer      PROP
      byte, dimension(:,:), allocatable :: pixq
 
      allocate(pixq(4096,4096))
 
c     print*,'ENTER read_wfc3uv_flt_smpl...'
 
      call read_wfc3uv_flt_full(FILENAME,pix,pixq,
     .                          CRPIX1, CRPIX2,
     .                          CRVAL1, CRVAL2,
     .                          CD1_1,  CD1_2,
     .                          CD2_1,  CD2_2,
     .                          LOFLAG, HIFLAG,
     .                          BDRY_XR,  BDRY_YR,
     .                          FILT,EXPT,
     .                          RDAT,PROP)
 
      deallocate(pixq)
 
c     print*,'EXIT  read_wfc3uv_flt_smpl...'
 
      return
      end
 
 
      subroutine read_wfc3uv_flt_full(FILENAME, pix, pxq,
     .                                CRPIX1, CRPIX2,
     .                                CRVAL1, CRVAL2,
     .                                CD1_1,  CD1_2,
     .                                CD2_1,  CD2_2,
     .                                LOFLAG, HIFLAG,
     .                                BDRY_XR,  BDRY_YR,
     .                                FILT,EXPT,
     .                                RDAT,PROP)
      implicit none
 
 
 
 
      character*80 FILENAME
      real pix(4096,4096)
      byte pxq(4096,4096)
 
      real*8    CRPIX1, CRPIX2
      real*8    CRVAL1, CRVAL2
      real*8    CD1_1,  CD1_2
      real*8    CD2_1,  CD2_2
 
      integer   LOFLAG, HIFLAG
      real*8    BDRY_XR(4,4)
      real*8    BDRY_YR(4,4)
 
      character*05 FILT
      real*4       EXPT
      real*4       RDAT
      integer      PROP
 
      integer NAXIS1
      integer NAXIS2
 
      real*8 COSPA, SINPA
      real*8 PA_APER
      real*8 PA_V3
 
      character*20 STREAM
      real*8  r8_query_hdre
      real*4  r4_query_hdre
      integer i4_query_hdre
      real*8  rdate_header
 
      integer i, j
      logical str_contains
 
      real*4, dimension(:,:),   allocatable :: pixz
 
 
c     print*,'   '
c     print*,'ENTER: subroutine read_wfc3uv_flt_full '
c     print*,'---> FILENAME: ',FILENAME
c     print*,'   '
 
      do i = 0001, 4096
      do j = 0001, 4096
         pix(i,j) = -750
         pxq(i,j) =    0
         enddo
         enddo
 
      NAXIS1 = r8_query_hdre(FILENAME,'NAXIS1  ',-1)
      NAXIS2 = r8_query_hdre(FILENAME,'NAXIS2  ',-1)
 
      CRPIX1 = r8_query_hdre(FILENAME,'CRPIX1  ',-1)
      CRPIX2 = r8_query_hdre(FILENAME,'CRPIX2  ',-1)
      CRVAL1 = r8_query_hdre(FILENAME,'CRVAL1  ',-1)
      CRVAL2 = r8_query_hdre(FILENAME,'CRVAL2  ',-1)
 
      PA_APER = r8_query_hdre(FILENAME,'PA_APER ',-1)
      PA_V3   = PA_APER - 0.0645919
      if (PA_V3.gt.360.0) PA_V3 = PA_V3 - 360.0
      if (PA_V3.lt.000.0) PA_V3 = PA_V3 + 360.0
      COSPA = cos(PA_V3*3.14159/180.0)
      SINPA = sin(PA_V3*3.14159/180.0)
 
      CD1_1 = -COSPA*0.039773/60/60
      CD1_2 =  SINPA*0.039773/60/60
      CD2_1 =  SINPA*0.039773/60/60
      CD2_2 =  COSPA*0.039773/60/60
 
      PROP = i4_query_hdre(FILENAME,'PROPOSID',-1)
      EXPT = r4_query_hdre(FILENAME,'EXPTIME ',-1)
      RDAT = rdate_header(FILENAME)
 
      call query_hdre(FILENAME,'FILTER  ',STREAM,-1)
 
      FILT = 'FXXXW'
      if (STREAM(1:1).eq.'F')  FILT = STREAM(1:5)
      if (STREAM(2:2).eq.'F')  FILT = STREAM(2:6)
 
      LOFLAG =   -50
      HIFLAG = 64000
 
      do i = 1, 4
      do j = 1, 4
         BDRY_XR(i,j) = 0.
         BDRY_YR(i,j) = 0.
         enddo
         enddo
 
      BDRY_XR(1,1) = 0001
      BDRY_YR(1,1) = 0001
      BDRY_XR(2,1) = 4095
      BDRY_YR(2,1) = 0001
      BDRY_XR(3,1) = 4095
      BDRY_YR(3,1) = 2047
      BDRY_XR(4,1) = 0001
      BDRY_YR(4,1) = 2047
 
      BDRY_XR(1,2) = 0001
      BDRY_YR(1,2) = 2049
      BDRY_XR(2,2) = 4095
      BDRY_YR(2,2) = 2049
      BDRY_XR(3,2) = 4095
      BDRY_YR(3,2) = 4095
      BDRY_XR(4,2) = 0001
      BDRY_YR(4,2) = 4095
 
      if (str_contains(FILENAME,80,'_flt.fits',09).or.
     .    str_contains(FILENAME,80,'_flc.fits',09)) then
         !print*,'call readfits_WFC4: ',FILENAME(1:40)
          call WFC3UV_FLTREAD(FILENAME,pix,pxq)
          goto 444
          endif
 
      if (str_contains(FILENAME,80,'_WJC.fits',09).or.
     .    str_contains(FILENAME,80,'_WJX.fits',09).or.
     .    str_contains(FILENAME,80,'_WXC.fits',09).or.
     .    str_contains(FILENAME,80,'_WJ2.fits',09)) then
          call readfits_j2r(FILENAME,pix(0001,0001),4096,4096)
          do i = 0001, 4096
          do j = 0001, 4096
             pxq(i,j) = 0
             if (pix(i,j).gt.HIFLAG) pxq(i,j) = 1
             enddo
             enddo
          goto 444
          endif
 
c
c raz type file...
c
      if (NAXIS1.eq.8412.and.NAXIS2.eq.2070) then
          allocate(pixz(8412,2070))
          call readfits_r4(FILENAME,pixz,8412,2070)
          do i = 0001, 2048
          do j = 0001, 2048
             pix(0000+i,0000+j) = pixz(0025+i,j)
             pix(2048+i,0000+j) = pixz(4177-i,j)
             pix(0000+i,4097-j) = pixz(4231+i,j+3)
             pix(2048+i,4097-j) = pixz(8383-i,j+3)
             enddo
             enddo
          deallocate(pixz)
          goto 444
          endif
 
      print*,'PROB : subroutine read_wfc3uv_flt_full '
      print*,'       could not read in file: '
      print*,'       ',FILENAME
      stop
 
 
  444 continue
      do i = 0001, 4096
      do j = 2046, 2051
         pix(i,j) = -750
         enddo
         enddo
 
c     print*,'   '
c     print*,'EXIT : subroutine read_wfc3uv_flt_full '
c     print*,'   '
 
      return
      end
 
 
 
 
 
 
 
 
 
 
 
c--------------------------------------------------------------
c
c check: UVIS-CENTER     --- full array --- ic9r16asq
c        UVIS1           --- udef sub   --- ichl08ujq
c        UVIS2           --- udef sub   --- ichl08ulq
c        UVIS2-M1K1C-SUB --- std sub    --- icdw04ljq_flt.fits
c        UVIS2-C1K1C-SUB --- std sub    --- icck01leq
c        UVIS2-C512C-SUB --- std sub    --- icjd04ezq_flt.fits
c        UVIS2-2K2C-SUB  --- std sub    --- icp601pyq_flt.fits
c        UVIS2-M512C-SUB --- std sub    --- ic8ea1eeq_flt.fits
c
c--------------------------------------------------------------
 
c--------------------------------------------
c
c
      subroutine WFC3UV_FLTREAD(FILE,pix,pxq)
      implicit none
      character*(*) FILE
      real pix(4096,4096)
      byte pxq(4096,4096)
 
      character*20 NAMEAP
      real pix0512(0512,0512)
      real pix0513(0513,0512)
      real pix1024(1024,1024)
      real pix1025(1025,1024)
      real pix4096(4096,2051)
      real pix2048(2048,2048)
      real pix2047(2047,2050)
      real pix4096g(4096,2050)
      integer i, j
      character*20 STREAM
 
      integer*2 pix0512i(0512,0512)
      integer*2 pix0513i(0513,0512)
      integer*2 pix1024i(1024,1024)
      integer*2 pix1025i(1025,1024)
      integer*2 pix4096i(4096,2051)
      integer*2 pix4096h(4096,2050)
      integer*2 pix2048i(2048,2048)
      integer*2 pix2047i(2047,2050)
 
 
 
      character*20 STREAM_CENTERA1
      character*20 STREAM_CENTERA2
      character*20 STREAM_SIZAXIS1
      character*20 STREAM_SIZAXIS2
      character*20 STREAM_NAXIS1
      character*20 STREAM_NAXIS2
      character*20 STREAM_BITPIX
      character*20 STREAM_CCDAMP
      character*20 STREAM_BINAXIS1
      character*20 STREAM_BINAXIS2
 
      integer CENTERA1
      integer CENTERA2
      integer SIZAXIS1
      integer SIZAXIS2
      integer NAXIS1
      integer NAXIS2
      integer BITPIX
      character*4 CCDAMP
      integer BINAXIS1
      integer BINAXIS2
 
c
c----------------------------------------------------------
c
 
      logical ISSUBARRAY
 
      ISSUBARRAY = .true.
 
      call query_hdre(FILE,'APERTURE',NAMEAP,-1)
      call query_hdre(FILE,'SUBARRAY',STREAM,-1)
      call query_hdre(FILE,'CENTERA1',STREAM_CENTERA1,-1)
      call query_hdre(FILE,'CENTERA2',STREAM_CENTERA2,-1)
      call query_hdre(FILE,'SIZAXIS1',STREAM_SIZAXIS1,-1)
      call query_hdre(FILE,'SIZAXIS2',STREAM_SIZAXIS2,-1)
      call query_hdre(FILE,'NAXIS1  ',STREAM_NAXIS1,-1)
      call query_hdre(FILE,'NAXIS2  ',STREAM_NAXIS2,-1)
      call query_hdre(FILE,'BITPIX  ',STREAM_BITPIX,-1)
      call query_hdre(FILE,'CCDAMP  ',STREAM_CCDAMP,-1)
 
      BINAXIS1 = 1
      BINAXIS2 = 1
      call query_hdre(FILE,'BINAXIS1',STREAM_BINAXIS1,-1)
      call query_hdre(FILE,'BINAXIS2',STREAM_BINAXIS2,-1)
 
      read(STREAM_CENTERA1,*) CENTERA1
      read(STREAM_CENTERA2,*) CENTERA2
      read(STREAM_SIZAXIS1,*) SIZAXIS1
      read(STREAM_SIZAXIS2,*) SIZAXIS2
      read(STREAM_NAXIS1  ,*) NAXIS1
      read(STREAM_NAXIS2  ,*) NAXIS2
      read(STREAM_BITPIX  ,*) BITPIX
      read(STREAM_BINAXIS1,*) BINAXIS1
      read(STREAM_BINAXIS2,*) BINAXIS2
 
      CCDAMP = STREAM_CCDAMP(2:5)
 
      do i = 0001, 4096
      do j = 0001, 4096
         pix(i,j) = -750
         enddo
         enddo
 
      if (BINAXIS1.ne.1.or.
     .    BINAXIS2.ne.1) then
          print*,'   '
          print*,'***********************************'
          print*,'*** '
          print*,'*** BINNED IMAGE ',FILE
          print*,'***    '
          print*,'*** DO NOT READ IN...'
          print*,'***    '
          print*,'***********************************'
          print*,'   '
          print*,'  STREAM_BINAXIS1:   ',BINAXIS1,STREAM_BINAXIS1
          print*,'  STREAM_BINAXIS2:   ',BINAXIS2,STREAM_BINAXIS2
          print*,'   '
          return
          endif
 
      if (NAMEAP(01:01).eq.'''') NAMEAP = NAMEAP(02:20)
      do i = 1, 20
         if (NAMEAP(i:i).eq.'''') NAMEAP(i:i) = ' '
         enddo
 
      read(STREAM,*) ISSUBARRAY
 
c     write(*,'(''WFC3UV_FLTREAD NAMEAP: '',20a,'' SUB? '',l1,5x,80a)')
c    .            NAMEAP,ISSUBARRAY,FILE
 
      if (.not.ISSUBARRAY) then
         if (NAMEAP(01:06).eq.'UVIS  '.or.
     .       NAMEAP(01:06).eq.'UVIS1 '.or.
     .       NAMEAP(01:06).eq.'UVIS2 '.or.
     .       NAMEAP(01:09).eq.'UVIS-FIX '.or.
     .       NAMEAP(01:09).eq.'UVIS1-FIX'.or.
     .       NAMEAP(01:09).eq.'UVIS2-FIX'.or.
     .       NAMEAP(01:09).eq.'G280-REF '.or.
     .       NAMEAP(01:09).eq.'UVIS-QUAD'.or.
     .       NAMEAP(01:11).eq.'UVIS-CENTER'.or.
     .       NAMEAP(01:11).eq.'UVIS-IR-FIX'.or.
     .       NAMEAP(01:12).eq.'UVIS1-IR-FIX'.or.
     .       NAMEAP(01:12).eq.'UVIS2-IR-FIX'.or.
     .       NAMEAP(01:15).eq.'UVIS1-C1K1C-CTE'.or.
     .       NAMEAP(01:15).eq.'UVIS2-C1K1C-CTE'.or.
     .       NAMEAP(01:13).eq.'UVIS-QUAD-FIX') then
             call readfits_WFC3(FILE,pix4096 ,1)
             call readfits_i2e(FILE,pix4096i,4096,2051,3)
             do i = 0001, 4096
             do j = 0001, 2048
                pix(i,j+0000) = pix4096(i,j)
                pxq(i,j+0000) = 0
                if (and(pix4096i(i,j),256).eq.256) pxq(i,j+0000) = 1
                enddo
                enddo
             call readfits_WFC3(FILE,pix4096,4)
             call readfits_i2e(FILE,pix4096i,4096,2051,6)
             do i = 0001, 4096
             do j = 0001, 2048
                pix(i,j+2048) = pix4096(i,j)
                pxq(i,j+2048) = 0
                if (and(pix4096i(i,j),256).eq.256) pxq(i,j+2048) = 1
                enddo
                enddo
             do i = 0001, 4096
             do j = 2048-2, 2049+2
                pix(i,j) = -750
                enddo
                enddo
             return
             endif
         print*,' '
         print*,'WFC3UV_FLTREAD not yet designed to operate on'
         print*,'          '
         print*,'FILENAME: ',FILE(1:40)
         print*,'          '
         print*,'aperture: ',NAMEAP
         print*,'subarray: ',ISSUBARRAY
         print*,'AS NON-SUBARRAY...'
         print*,' '
         do i = 0001, 4096
         do j = 0001, 4096
            pix(i,j) = -1000
            pxq(i,j) = 0
            pxq(i,j) = 1
            enddo
            enddo
         return
         endif
 
      if (NAMEAP(01:15).eq.'UVIS1-C512A-SUB') then
          call readfits_r4e(FILE,pix0513 ,0513,0512,1)
          call readfits_i2e(FILE,pix0513i,0513,0512,3)
          do i = 0001, 0513
          do j = 0001, 0512-2
             pix(i+0000,j+1538+2048) = pix0513(i,j)
             enddo
             enddo
          do i = 0001, 0513
          do j = 0002, 0511-2
             pxq(i+0000,j+1538+2048) = 0
	     if (iand(pix0513i(i,j),256).eq.256) then
                pxq(i+0000,j+1538+2048) = 1
                endif
             enddo
             enddo
          return
          endif
 
      if (NAMEAP(01:15).eq.'UVIS2-C1K1C-SUB') then
         !print*,'C1K1Ca: ',FILE
          call readfits_r4e(FILE,pix1025 ,1025,1024,1)
          call readfits_i2e(FILE,pix1025i,1025,1024,3)
          do i = 0001, 1025
          do j = 0001, 1024
             pix(i+0000,j+0001) = pix1025(i,j)
             pxq(i+0000,j+0001) = 0
	     if (iand(pix1025i(i,j),256).eq.256) then
                pxq(i+0000,j+0001) = 1
                endif
             enddo
             enddo
         !call writfits_r4('pix1025.fits' ,pix1025 ,1025,1024)
         !call writfits_i2('pix1025i.fits',pix1025i,1025,1024)
         !print*,'C1K1Cb: ',FILE
          return
          endif
 
      if (NAMEAP(01:15).eq.'UVIS2-M1K1C-SUB') then
          call readfits_r4e(FILE,pix1024 ,1024,1024,1)
          call readfits_i2e(FILE,pix1024i,1024,1024,3)
          do i = 0001, 1024
          do j = 0001, 1024
             pix(i+1023,j+1027) = pix1024(i,j)
             pxq(i+1023,j+1027) = 0
	     if (iand(pix1024i(i,j),256).eq.256) then
                      pxq(i+1023,j+1027) = 1
                endif
             enddo
             enddo
          return
          endif
 
      if (NAMEAP(01:15).eq.'UVIS2-C512C-SUB') then
          call readfits_r4e(FILE,pix0513 ,0513,0512,1)
          call readfits_i2e(FILE,pix0513i,0513,0512,3)
          do i = 0001, 0513
          do j = 0001, 0512
             pix(i+0000,j+0001) = pix0513(i,j)
             pxq(i+0000,j+0001) = 0
	     if (iand(pix0513i(i,j),256).eq.256) then
                pxq(i+0000,j+0001) = 1
                endif
             enddo
             enddo
          return
          endif
 
      if (NAMEAP(01:15).eq.'UVIS2-C512D-SUB') then
          call readfits_r4e(FILE,pix0513 ,0513,0512,1)
          call readfits_i2e(FILE,pix0513i,0513,0512,3)
          do i = 0001, 0513
          do j = 0001, 0512
             pix(i+3583,j+0001) = pix0513(i,j)
             pxq(i+3583,j+0001) = 0
	     if (iand(pix0513i(i,j),256).eq.256) then
                pxq(i+3583,j+0001) = 1
                endif
             enddo
             enddo
          return
          endif
 
      if (NAMEAP(01:15).eq.'UVIS1-C512B-SUB') then
          call readfits_r4e(FILE,pix0513 ,0513,0512,1)
          call readfits_i2e(FILE,pix0513i,0513,0512,3)
          do i = 0001, 0513
          do j = 0001, 0512
             pix(i+3583,j+1538) = pix0513(i,j)
             pxq(i+3583,j+1538) = 0
	     if (iand(pix0513i(i,j),256).eq.256) then
                pxq(i+3583,j+1538) = 1
                endif
             enddo
             enddo
          return
          endif
 
      if (NAMEAP(01:14).eq.'UVIS2-M512-SUB') then
          call readfits_r4e(FILE,pix0512 ,0512,0512,1)
          call readfits_i2e(FILE,pix0512i,0512,0512,3)
          do i = 0001, 0512
          do j = 0001, 0512
             pix(i+1791,j+1539) = pix0512(i,j)
             pxq(i+1791,j+1539) = 0
	     if (iand(pix0512i(i,j),256).eq.256) then
                pxq(i+1791,j+1539) = 1
                endif
             enddo
             enddo
          return
          endif
 
      if (NAMEAP(01:15).eq.'UVIS2-M512C-SUB') then
          call readfits_r4e(FILE,pix0512 ,0512,0512,1)
          call readfits_i2e(FILE,pix0512i,0512,0512,3)
          do i = 0001, 0512
          do j = 0001, 0512
             pix(i+1535,j+1539) = pix0512(i,j)
             pxq(i+1535,j+1539) = 0
	     if (iand(pix0512i(i,j),256).eq.256) then
                pxq(i+1535,j+1539) = 1
                endif
             enddo
             enddo
          return
          endif
 
      if (NAMEAP(01:14).eq.'UVIS1-M512-SUB') then
          call readfits_r4e(FILE,pix0512 ,0512,0512,1)
          call readfits_i2e(FILE,pix0512i,0512,0512,3)
          do i = 0001, 0512
          do j = 0001, 0512
             pix(i+1791,j+2048) = pix0512(i,j)
             pxq(i+1791,j+2048) = 0
	     if (iand(pix0512i(i,j),256).eq.256) then
                pxq(i+1791,j+2048) = 1
                endif
             enddo
             enddo
          return
          endif
 
 
      if (NAMEAP(01:15).eq.'UVIS2-2K2C-SUB ') then
          call readfits_r4e(FILE,pix2047 ,2047,2050,1)
          call readfits_i2e(FILE,pix2047i,2047,2050,3)
          do i = 0001, 2047
          do j = 0001, 2050
             pix(i+0000,j+0001) = pix2047(i,j)
             pxq(i+0000,j+0001) = 0
	     if (iand(pix2047i(i,j),256).eq.256) then
                pxq(i+0000,j+0001) = 1
                endif
             enddo
             enddo
          return
          endif
 
      if (NAMEAP(01:15).eq.'UVIS2-2K2D-SUB ') then
          call readfits_r4e(FILE,pix2047 ,2047,2050,1)
          call readfits_i2e(FILE,pix2047i,2047,2050,3)
          do i = 0001, 2047
          do j = 0001, 2050
             pix(i+2049,j+0001) = pix2047(i,j)
             pxq(i+2049,j+0001) = 0
	     if (iand(pix2047i(i,j),256).eq.256) then
                pxq(i+2049,j+0001) = 1
                endif
             enddo
             enddo
          return
          endif
 
 
      if (NAMEAP(01:15).eq.'UVIS1-2K2B-SUB ') then
c         print*,'read1...'
          call readfits_r4e(FILE,pix2047 ,2047,2050,1)
c         print*,'read3...'
          call readfits_i2e(FILE,pix2047i,2047,2050,3)
          do i = 0001, 2047
          do j = 0001, 2048
             pix(i+2049,j+2048) = pix2047(i,j)
             pxq(i+2049,j+2048) = 0
	     if (iand(pix2047i(i,j),256).eq.256) then
                pxq(i+2049,j+2048) = 1
                endif
             enddo
             enddo
          return
          endif
 
 
      if (NAMEAP(01:15).eq.'UVIS1-2K2A-SUB ') then
          call readfits_r4e(FILE,pix2047 ,2047,2050,1)
          call readfits_i2e(FILE,pix2047i,2047,2050,3)
          do i = 0001, 2047
          do j = 0001, 2048
             pix(i+0000,j+2048) = pix2047(i,j)
             pxq(i+0000,j+2048) = 0
	     if (iand(pix2047i(i,j),256).eq.256) then
                pxq(i+0000,j+2048) = 1
                endif
             enddo
             enddo
          return
          endif
 
 
      if (NAMEAP(01:15).eq.'UVIS2          ') then
         print*,'UVIS2... user defined subarray...'
         print*,'UVIS2... really!...'
         print*,'CENTERA1 = ',CENTERA1
         print*,'CENTERA2 = ',CENTERA2
         print*,'SIZAXIS1 = ',SIZAXIS1
         print*,'SIZAXIS2 = ',SIZAXIS2
         do i = 0001, 4096
         do j = 0001, 4096
            pix(i,j) = 0.00
            enddo
            enddo
         if (CENTERA1.eq.1557.and.
     .       CENTERA2.eq.1540.and.
     .       SIZAXIS1.eq.1024.and.
     .       SIZAXIS2.eq.1024) then
             call readfits_r4e(FILE,pix1024 ,1024,1024,1)
             call readfits_i2e(FILE,pix1024i,1024,1024,3)
             do i = 0001, 1024
             do j = 0001, 1024
                pix(i+1020-1,j+1028-1) = pix1024(i,j)
                pxq(i+1020-1,j+1028-1) = 0
	        if (iand(pix1024i(i,j),256).eq.256) then
                   pxq(i+1020-1,j+1028-1) = 1
                   endif
                enddo
                enddo
             return
             endif
         if (CENTERA2.ne.1027) stop 'CENTERA2.ne.1027'
         if (SIZAXIS1.ne.2048) stop 'SIZAXIS1.ne.2048'
         if (SIZAXIS2.ne.2048) stop 'SIZAXIS2.ne.2048'
         if (CCDAMP.ne.'C   ') stop 'CCDAMP.ne."C   "'
         call readfits_r4e(FILE,pix2048 ,2048,2048,1)
         call readfits_i2e(FILE,pix2048i,2048,2048,3)
         do i = 0001, 2048
         do j = 0001, 2048
            pix(i+1024-1,j+0003-1) = pix2048(i,j)
            pxq(i+1024-1,j+0003-1) = 0
	    if (iand(pix2048i(i,j),256).eq.256) then
                pxq(i+1024-1,j+0003-1) = 1
                endif
            enddo
            enddo
         return
         endif
 
 
      if (NAMEAP(01:15).eq.'UVIS1          ') then
         print*,'UVIS1... user defined subarray...'
         if (CENTERA1.ne.2073) stop 'CENTERA1.ne.2073'
         if (CENTERA2.ne.1027) stop 'CENTERA2.ne.1027'
         if (SIZAXIS1.ne.2048) stop 'SIZAXIS1.ne.2048'
         if (SIZAXIS2.ne.2048) stop 'SIZAXIS2.ne.2048'
         if (CCDAMP.ne.'A   ') stop 'CCDAMP.ne."A   "'
         print*,' FILE: ',FILE
         print*,'CALL readfits_r4e...'
         call readfits_r4e(FILE,pix2048 ,2048,2048,1)
         call readfits_i2e(FILE,pix2048i,2048,2048,3)
         print*,'RETN readfits_r4e...'
         print*,'LOOP1...'
         print*,'LOOP2...'
         do i = 0001, 2048
         do j = 0001, 2048
            if (i+1024-0001.ge.0001.and.i+1024-0001.le.4096.and.
     .          j+0003+2048.ge.0001.and.j+0003+2048.le.4096) then
                pix(i+1024-1,j+0003+2048-1) = pix2048(i,j)
                pxq(i+1024-1,j+0003+2048-1) = 0
	        if (iand(pix2048i(i,j),256).eq.256) then
                    pxq(i+1024-1,j+0003+2048-1) = 1
                    endif
                endif
            enddo
            enddo
         print*,'LOOP3...'
         print*,'RETN'
         return
         endif
 
      if (NAMEAP(01:15).eq.'UVIS2-2K4-SUB  ') then
         print*,'UVIS2-2K4-SUB... is it workingg? (jay?)'
         if (CENTERA1.ne.2074) stop 'CENTERA1.ne.2074'
         if (CENTERA2.ne.1027) stop 'CENTERA2.ne.1027'
         if (CCDAMP.ne.'D   ') stop 'CCDAMP.ne."D   "'
         print*,'FILE: ',FILE
         print*,'CALL readfits_r4e...'
         call readfits_r4e(FILE,pix4096g,4096,2050,1)
         print*,'CALL readfits_i2e...'
         call readfits_i2e(FILE,pix4096h,4096,2050,3)
         print*,'RETN readfits_r4e...'
         print*,'LOOP1...'
         do i = 0001, 4096
         do j = 0001, 2048
            pix(i,j) = pix4096g(i,j)
            pxq(i,j) = 0
	    if (iand(pix4096h(i,j),256).eq.256) pxq(i,j) = 1
            enddo
            enddo
c        print*,'OUTPUT TEMP...'
c        call writfits_r4('temp.fits',pix,4096,4096)
c        if (.true.) stop
         return
         endif
 
 
      if (NAMEAP(01:15).eq.'UVIS1-2K4-SUB  ') then
         print*,'UVIS1-2K4-SUB... is it working? (jay?)'
c        if (CENTERA1.ne.2074) stop 'CENTERA1.ne.2074'
c        if (CENTERA2.ne.1027) stop 'CENTERA2.ne.1027'
c        if (CCDAMP.ne.'D   ') stop 'CCDAMP.ne."D   "'
         print*,'FILE: ',FILE
         print*,'CALL readfits_r4e...'
         call readfits_r4e(FILE,pix4096g,4096,2050,1)
         call readfits_i2e(FILE,pix4096h,4096,2050,3)
         print*,'RETN readfits_r4e...'
         print*,'LOOP1...'
         do i = 0001, 4096
         do j = 0001, 2048
            pix(i,j+2048) = pix4096g(i,j)
            pxq(i,j+2048) = 0
            if (iand(pix4096h(i,j),256).eq.256) pxq(i,j+2048) = 1
            enddo
            enddo
c        print*,'OUTPUT TEMP...'
c        call writfits_r4('temp.fits',pix,4096,4096)
         return
         endif
 
 4444 continue
 
      do i = 0001, 4096
      do j = 0001, 4096
         pix(i,j) = -1000
         pxq(i,j) = 0
         pxq(i,j) = 1
         enddo
         enddo
 
      print*,' '
      print*,'4444 '
      print*,' '
      print*,'WFC3UV_FLTREAD not yet designed to operate on'
      print*,'          '
      print*,'    FILE: ',FILE(1:80)
      print*,'          '
      print*,'aperture: ',NAMEAP
      print*,'subarray: ',ISSUBARRAY
      print*,'AS A SUBARRAY...'
      print*,' '
      return
 
      end
 
 
 
 
 
c--------------------------------------------
c
c
      subroutine readfits_WFC3(FILE,pix,nimg)
      implicit none
 
      character*80 FILE
      real pix(4096,2051)
 
      character*70 INFO(10)
      common / fitsinfo / INFO
 
      integer nimg
      integer naxes
      integer laxis(3)
 
      character*8  field
      character*20 stream
 
      integer nbyte0
      integer nbyteE
      integer nbyte1
      integer nbyte2
      integer nbper
      integer i,ios, k
 
      character*2880 buffc
      byte buffb(2880)
      equivalence (buffc,buffb)
 
      integer ifirst, i1, i2
 
      integer np1, np2, npt
      integer nextend
      integer nread
      real bscale, bzero
      integer bitpix
      logical islinux, islinuxu
 
 
      character*70 HDR(25)
      common/HDR/HDR
 
      open(10,file=FILE,status='old',
     .     err=900,recl=2880,form='UNFORMATTED',
     .     access='DIRECT')
 
 
      bscale = 1.0
      bzero  = 0.0
 
      naxes = -1
      laxis(1) = 1
      laxis(2) = 1
      laxis(3) = 1
      nextend = 0
 
      do i = 1, 10
         INFO(i) = ' '
         enddo
      do i = 1, 25
         HDR(i) = ' '
         enddo
 
      i = 0
      nread = 0
 100  continue
      i = i + 1
      read(10,rec=i,iostat=ios) buffc
      do k = 0, 35, 1
         field  = buffc(k*80+01:k*80+08)
         stream = buffc(k*80+11:k*80+31)
         if (field.eq.'NAXIS   ') read(stream,*) naxes
         if (field.eq.'NAXIS1  ') read(stream,*) laxis(1)
         if (field.eq.'NAXIS2  ') read(stream,*) laxis(2)
         if (field.eq.'NAXIS3  ') read(stream,*) laxis(3)
         if (field.eq.'NEXTEND ') read(stream,*) nextend
         if (field.eq.'BITPIX  ') read(stream,*) bitpix
         if (field.eq.'BSCALE  ') read(stream,*) bscale
         if (field.eq.'BZERO   ') read(stream,*) bzero
 
         if (field.eq.'EXPTIME ') INFO(01) = stream
         if (field.eq.'FILTER  ') INFO(02) = stream
         if (field.eq.'FILENAME') INFO(03) = stream
         if (field.eq.'DATE-OBS') INFO(04) = stream
         if (field.eq.'TIME-OBS') INFO(05) = stream
         if (field.eq.'DEC_TARG') INFO(06) = stream
         if (field.eq.'RA_TARG ') INFO(07) = stream
         if (field.eq.'PA_V3   ') INFO(08) = stream
         if (field.eq.'PROPOSID') INFO(09) = stream
         if (field.eq.'CCDGAIN ') INFO(10) = stream
 
         if (field.eq.'CRPIX1  ') HDR(01) = stream
         if (field.eq.'CRPIX2  ') HDR(02) = stream
         if (field.eq.'CRVAL1  ') HDR(03) = stream
         if (field.eq.'CRVAL2  ') HDR(04) = stream
         if (field.eq.'CTYPE1  ') HDR(05) = stream
         if (field.eq.'CTYPE2  ') HDR(06) = stream
         if (field.eq.'CD1_1   ') HDR(07) = stream
         if (field.eq.'CD1_2   ') HDR(08) = stream
         if (field.eq.'CD2_1   ') HDR(09) = stream
         if (field.eq.'CD2_2   ') HDR(10) = stream
         if (field.eq.'ORIENTAT') HDR(11) = stream
         if (field.eq.'PA_APER ') HDR(12) = stream
         if (field.eq.'PA_V3   ') HDR(13) = stream
         if (field.eq.'DATE-OBS') HDR(14) = stream
         if (field.eq.'TIME-OBS') HDR(15) = stream
         if (field.eq.'EXPTIME ') HDR(16) = stream
         if (field.eq.'ROOTNAME') HDR(17) = stream
         if (field.eq.'TARGNAME') HDR(18) = stream
         if (field.eq.'RA_TARG ') HDR(19) = stream
         if (field.eq.'DEC_TARG') HDR(20) = stream
         if (field.eq.'PROPOSID') HDR(21) = stream
         if (field.eq.'FILTER  ') HDR(22) = stream
         if (field.eq.'FILTER  ') HDR(23) = stream
         if (field.eq.'VAFACTOR') HDR(24) = stream
         if (field.eq.'CCDGAIN ') HDR(25) = stream
         if (field.eq.'END     ') goto 101
         enddo
      goto 100
 101  continue
 
 
      nread = nread + 1
 
      ifirst = i+1
      i1 = i
      i2 = i
 
      if (naxes.eq.0) then    ! maybe multiple images stored as extensions
         if (nextend.eq.0) then
            print*,'THIS IS A NULL IMAGE: '
            print*,'NAXES: ',NAXES
            print*,'NEXND: ',NEXTEND
            stop
            endif
         endif
 
      if (nread.ne.nimg+1) then
         nbper  = abs(BITPIX/8)*laxis(1)*laxis(2)
         if (NAXES.eq.0) nbper = 0
         i = i + 1.0*nbper/2880 + 0.9999
         goto 100
         endif
 
      if (laxis(1).ne.4096.or.
     .    laxis(2).ne.2051) then
          print*,'  laxis1: ',laxis(1)
          print*,'  laxis2: ',laxis(2)
          print*,'    4096: ',4096
          print*,'    2051: ',2051
          stop
          endif
 
      if (naxes.eq.2) then   ! nimg is irrelevant; ignore
         nbper  = 4*laxis(1)*laxis(2)
         npt    = laxis(1)*laxis(2)
         nbyte1 = 1
         nbyte2 = nbper
         i1 = i+1 + nbyte1/2880
         i2 = i+1 + nbyte2/2880
         endif
 
      if (naxes.eq.3) then
         nbper  = 4*laxis(1)*laxis(2)
         nbyte1 = 1 + nbper*(nimg-1)
         nbyte2 =     nbper*(nimg  )
         i1 = i+1 + nbyte1/2880
         i2 = i+1 + nbyte2/2880
         npt = laxis(1)*laxis(2)
         endif
 
      islinuxu = islinux()
      do i = i1, i2, 1
         read(10,rec=i,iostat=ios) buffc
         nbyte0 = (i-ifirst)*2880+   1
         nbyteE = (i-ifirst)*2880+2880
         np1 = (nbyte0-nbyte1)/4 + 1
         np2 = (nbyteE-nbyte1)/4 + 1
         call buff2pix_r4q(buffb,pix,np1,npt,islinuxu)
         enddo
 
      return
 
 900  continue
      print*,'readfits_WFC3: READFITS ERROR'
      stop
 
      end
 
 
      subroutine buff2pix_r4q(buff,pix,n1,nt,islinuxu)
      implicit none
      byte buff(2880)
      real pix(*)
      integer n1,nt
      logical islinuxu
 
      byte b(4)
      real r
      equivalence(r,b)
 
      integer i, npu, nbu
 
      do i = 1, 720
         npu = n1+i-1
         nbu = (i-1)*4
         if (.not.islinuxu) then
            b(1) = buff(nbu+1)
            b(2) = buff(nbu+2)
            b(3) = buff(nbu+3)
            b(4) = buff(nbu+4)
            endif
         if (islinuxu) then
            b(4) = buff(nbu+1)
            b(3) = buff(nbu+2)
            b(2) = buff(nbu+3)
            b(1) = buff(nbu+4)
            endif
         if (npu.ge.1.and.npu.le.nt) pix(npu) = r
         enddo
 
      return
      end
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/FITSIO/WFPC2/read_wfpc2_flt_full.f"
c****                                         
c*********************************************
                                              
 
      subroutine read_wfpc2_flt_full(FILENAME,pix,
     .                               CRPIX1, CRPIX2,
     .                               CRVAL1, CRVAL2,
     .                               CD1_1,  CD1_2,
     .                               CD2_1,  CD2_2,
     .                               LOFLAG, HIFLAG,
     .                               BDRY_XR,  BDRY_YR,
     .                               FILT,EXPT,
     .                               RDAT,PROP)
      implicit none
 
 
 
 
      character*80 FILENAME
      real      pix(1600,1600)
 
      real*8    CRPIX1, CRPIX2
      real*8    CRVAL1, CRVAL2
      real*8    CD1_1,  CD1_2
      real*8    CD2_1,  CD2_2
 
      integer   LOFLAG, HIFLAG
      real*8    BDRY_XR(4,4)
      real*8    BDRY_YR(4,4)
 
      character*05 FILT
      real*4       EXPT
      real*4       RDAT
      integer      PROP
 
      integer NAXIS1
      integer NAXIS2
 
      real*8 COSPA, SINPA
      real*8 PA_V3
 
      real   rGAIN
      character*80 FILTU
      character*80 FILT3
      character*80 FILT4
      real*8  r8_query_hdre
      real*4  r4_query_hdre
      integer i4_query_hdre
      real*8  rdate_header
 
      integer i, j
      integer k
 
      real*4, dimension(:,:,:), allocatable :: pix_3D
      character*03 TYP
      character*80 STREAM
 
 
      allocate(pix_3d(800,800,4))
 
      HIFLAG = 25000
      LOFLAG = -50.0
      print*,' '
      print*,'read in pix file...',FILENAME
      print*,' '
 
      TYP = 'XXX'
      do i = 1, 75
         if (FILENAME(i:i+8).eq.'_c0f.fits') TYP = 'c0f'
         if (FILENAME(i:i+8).eq.'_c0m.fits') TYP = 'c0m'
         enddo
      if (TYP.eq.'c0f') then
          print*,'readfits_r4_3D...',FILENAME
          call readfits_r4_3D(FILENAME,pix_3D,800,800,4)
          endif
      if (TYP.eq.'c0m') then
          print*,'readfits_r4e...',FILENAME
          call readfits_r4e(FILENAME,pix_3D(1,1,1),800,800,1)
          call readfits_r4e(FILENAME,pix_3D(1,1,2),800,800,2)
          call readfits_r4e(FILENAME,pix_3D(1,1,3),800,800,3)
          call readfits_r4e(FILENAME,pix_3D(1,1,4),800,800,4)
          endif
 
      NAXIS1 = 1600
      NAXIS2 = 1600
 
      rGAIN = r4_query_hdre(FILENAME,'ATODGAIN',-1)
 
      do i = 001, 800
      do j = 001, 800
         if (i.lt.045) pix_3D(i,j,1) = -999.9
         if (j.lt.055) pix_3D(i,j,1) = -999.9
         if (i.lt.050) pix_3D(i,j,2) = -999.9
         if (j.lt.025) pix_3D(i,j,2) = -999.9
         if (i.lt.030) pix_3D(i,j,3) = -999.9
         if (j.lt.045) pix_3D(i,j,3) = -999.9
         if (i.lt.045) pix_3D(i,j,4) = -999.9
         if (j.lt.045) pix_3D(i,j,4) = -999.9
         if (i.gt.797) then
            pix_3D(i,j,1) = -999.9
            pix_3D(i,j,2) = -999.9
            pix_3D(i,j,3) = -999.9
            pix_3D(i,j,4) = -999.9
            endif
         if (j.gt.797) then
            pix_3D(i,j,1) = -999.9
            pix_3D(i,j,2) = -999.9
            pix_3D(i,j,3) = -999.9
            pix_3D(i,j,4) = -999.9
            endif
         pix(i+000,j+000) = pix_3D(i,j,1)*rGAIN
         pix(i+800,j+000) = pix_3D(i,j,2)*rGAIN
         pix(i+800,j+800) = pix_3D(i,j,3)*rGAIN
         pix(i+000,j+800) = pix_3D(i,j,4)*rGAIN
         enddo
         enddo
      deallocate(pix_3D)
 
      CRPIX1 = r8_query_hdre(FILENAME,'CRPIX1  ',-1)
      CRPIX2 = r8_query_hdre(FILENAME,'CRPIX2  ',-1)
      CRVAL1 = r8_query_hdre(FILENAME,'CRVAL1  ',-1)
      CRVAL2 = r8_query_hdre(FILENAME,'CRVAL2  ',-1)
 
      PA_V3 = r4_query_hdre(FILENAME,'PA_V3   ',-1)
      PA_V3 = PA_V3 - 0.1884335 - 315.0 - 0.321
      CD1_1 = -2.77778e-05*cos(PA_V3*3.14159/180.0)/1.003
      CD2_1 =  2.77778e-05*sin(PA_V3*3.14159/180.0)/1.003
      CD1_2 =  2.77778e-05*sin(PA_V3*3.14159/180.0)/1.003
      CD2_2 =  2.77778e-05*cos(PA_V3*3.14159/180.0)/1.003
 
      PROP = i4_query_hdre(FILENAME,'PROPOSID',-1)
      EXPT = r4_query_hdre(FILENAME,'EXPTIME ',-1)
      RDAT = rdate_header(FILENAME)
 
      call query_hdre(FILENAME,'FILTER  ',FILTU,-1)
      call query_hdre(FILENAME,'FILTNAM1',FILT3,-1)
      call query_hdre(FILENAME,'FILTNAM2',FILT4,-1)
 
      if (FILTU(1:1).eq.'F') FILT = FILTU(1:5)
      if (FILTU(2:2).eq.'F') FILT = FILTU(2:6)
      if (FILT3(1:1).eq.'F') FILT = FILT3(1:5)
      if (FILT3(2:2).eq.'F') FILT = FILT3(2:6)
      if (FILT4(1:1).eq.'F') FILT = FILT4(1:5)
      if (FILT4(2:2).eq.'F') FILT = FILT4(2:6)
 
      BDRY_XR(1,1) = 0075
      BDRY_YR(1,1) = 0075
      BDRY_XR(2,1) = 0797
      BDRY_YR(2,1) = 0075
      BDRY_XR(3,1) = 0797
      BDRY_YR(3,1) = 0797
      BDRY_XR(4,1) = 0075
      BDRY_YR(4,1) = 0797
 
      BDRY_XR(1,2) = 0075 + 800
      BDRY_YR(1,2) = 0075
      BDRY_XR(2,2) = 0797 + 800
      BDRY_YR(2,2) = 0075
      BDRY_XR(3,2) = 0797 + 800
      BDRY_YR(3,2) = 0797
      BDRY_XR(4,2) = 0075 + 800
      BDRY_YR(4,2) = 0797
 
      BDRY_XR(1,3) = 0075 + 800
      BDRY_YR(1,3) = 0075 + 800
      BDRY_XR(2,3) = 0797 + 800
      BDRY_YR(2,3) = 0075 + 800
      BDRY_XR(3,3) = 0797 + 800
      BDRY_YR(3,3) = 0797 + 800
      BDRY_XR(4,3) = 0075 + 800
      BDRY_YR(4,3) = 0797 + 800
 
      BDRY_XR(1,4) = 0075
      BDRY_YR(1,4) = 0075 + 800
      BDRY_XR(2,4) = 0797
      BDRY_YR(2,4) = 0075 + 800
      BDRY_XR(3,4) = 0797
      BDRY_YR(3,4) = 0797 + 800
      BDRY_XR(4,4) = 0075
      BDRY_YR(4,4) = 0797 + 800
 
      return
 
      end
 
 
c      COSPA = cos(PA_V3*3.14159/180.0)
c      SINPA = sin(PA_V3*3.14159/180.0)
c
c      CD1_1 = -COSPA*0.049730/60/60
c      CD1_2 =  SINPA*0.049730/60/60
c      CD2_1 =  SINPA*0.049730/60/60
c      CD2_2 =  COSPA*0.049730/60/60
c
c      call query_hdre(FILENAME,'FILTER  ',FILT )
c      call query_hdre(FILENAME,'FILTER1 ',FILT1)
c      call query_hdre(FILENAME,'FILTER2 ',FILT2)
c      call query_hdre(FILENAME,'FILTNAM1',FILT3)
c      call query_hdre(FILENAME,'FILTNAM2',FILT4)
c
c      if (FILT1(1:1).eq.'F')  FILT = FILT1(1:5)
c      if (FILT2(1:1).eq.'F')  FILT = FILT2(1:5)
c      if (FILT3(1:1).eq.'F')  FILT = FILT3(1:5)
c      if (FILT4(1:1).eq.'F')  FILT = FILT4(1:5)
c      if (FILT1(2:2).eq.'F')  FILT = FILT1(2:6)
c      if (FILT2(2:2).eq.'F')  FILT = FILT2(2:6)
c      if (FILT3(2:2).eq.'F')  FILT = FILT3(2:6)
c      if (FILT4(2:2).eq.'F')  FILT = FILT4(2:6)
c
c      LOFLAG =   -50
c      HIFLAG = 60000
c
c      do i = 1, 4
c      do j = 1, 4
c         BDRY_XR(i,j) = 0.
c         BDRY_YR(i,j) = 0.
c         enddo
c         enddo
c
c      BDRY_XR(1,1) = 0001
c      BDRY_YR(1,1) = 0001
c      BDRY_XR(2,1) = 4095
c      BDRY_YR(2,1) = 0001
c      BDRY_XR(3,1) = 4095
c      BDRY_YR(3,1) = 2047
c      BDRY_XR(4,1) = 0001
c      BDRY_YR(4,1) = 2047
c
c      BDRY_XR(1,2) = 0001
c      BDRY_YR(1,2) = 2049
c      BDRY_XR(2,2) = 4095
c      BDRY_YR(2,2) = 2049
c      BDRY_XR(3,2) = 4095
c      BDRY_YR(3,2) = 4095
c      BDRY_XR(4,2) = 0001
c      BDRY_YR(4,2) = 4095
c
c
c      call readfits_WFC(FILENAME,pix(0001,2049),4)
c      call readfits_WFC(FILENAME,pix(0001,0001),1)
c
c      do i = 0001, 4096
c      do j = 2046, 2051
c         pix(i,j) = -750
c         enddo
c         enddo
c
c      return
c      end
c
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/FITSIO/HSTDRZ/read_hstdrz_full.f"
c****                                         
c*********************************************
                                              
 
 
c*********************************************
c****
c**** #include "/user/jayander/FORTRAN/FITSIO/HSTDRZ/read_hstdrz_full.f"
c****
c*********************************************
 
 
 
      subroutine read_hstdrz_full(FILENAME,
     .                            pix,NAXIS1,NAXIS2,
     .                            CRPIX1,   CRPIX2,
     .                            CRVAL1,   CRVAL2,
     .                            CD1_1,    CD1_2,
     .                            CD2_1,    CD2_2,
     .                            LOFLAG,   HIFLAG,
     .                            BDRY_XR,  BDRY_YR,
     .                            FILT,     EXPT,
     .                            RDAT,     PROP)
      implicit none
 
      character*80 FILENAME
      real         pix(NAXIS1,NAXIS2)
      integer      NAXIS1
      integer      NAXIS2
 
      real*8       CRPIX1, CRPIX2
      real*8       CRVAL1, CRVAL2
      real*8       CD1_1,  CD1_2
      real*8       CD2_1,  CD2_2
 
      integer      LOFLAG, HIFLAG
      real*8       BDRY_XR(4,4)
      real*8       BDRY_YR(4,4)
 
      character*05 FILT
      real*4       EXPT
      real*4       RDAT
      integer      PROP
 
      character*80 FILENAMEu
 
      byte            , dimension(:,:), allocatable :: sat
      integer (kind=4), dimension(:,:), allocatable :: pxn
      real    (kind=4), dimension(:,:), allocatable :: pxf
      real    (kind=4), dimension(:,:), allocatable :: pxg
      byte            , dimension(:,:), allocatable :: uuu
 
      integer Ns
      integer i, j, k, kk, kkx
      integer i_k(4), j_k(4)
      real    t_k(4)
 
      real*8       rdate_header
      integer*4    i4_query_hdre
      real*4       r4_query_hdre
      real*8       r8_query_hdre
      character*20 STREAM
 
      integer ii, jj
 
      real DMAX, dx, dy, dd, tt, atan2
 
      allocate(sat(NAXIS1,NAXIS2))
      allocate(pxn(NAXIS1,NAXIS2))
      allocate(pxf(NAXIS1,NAXIS2))
      allocate(pxg(NAXIS1,NAXIS2))
      allocate(uuu(NAXIS1,NAXIS2))
 
      LOFLAG = -50
      HIFLAG = 1e9
 
      call query_hdre(FILENAME,'EXTEND  ',STREAM,-1)
 
      if (STREAM(1:4).eq.'NULL') then
          print*,'readfits_r4e[0]: ',FILENAME(1:30)
          call readfits_r4e(FILENAME,pix,NAXIS1,NAXIS2,0)
          CRPIX1 = r8_query_hdre(FILENAME,'CRPIX1  ',-1)
          CRPIX2 = r8_query_hdre(FILENAME,'CRPIX2  ',-1)
          CRVAL1 = r8_query_hdre(FILENAME,'CRVAL1  ',-1)
          CRVAL2 = r8_query_hdre(FILENAME,'CRVAL2  ',-1)
          CD1_1  = r8_query_hdre(FILENAME,'CD1_1   ',-1)
          CD1_2  = r8_query_hdre(FILENAME,'CD1_2   ',-1)
          CD2_1  = r8_query_hdre(FILENAME,'CD2_1   ',-1)
          CD2_2  = r8_query_hdre(FILENAME,'CD2_2   ',-1)
          BDRY_XR(k,1) = 1
          BDRY_YR(k,1) = 1
          BDRY_XR(2,1) = 1
          BDRY_YR(2,1) = NAXIS2
          BDRY_XR(3,1) = NAXIS1
          BDRY_YR(3,1) = NAXIS2
          BDRY_XR(4,1) = NAXIS1
          BDRY_YR(4,1) = 1
          return
         else
          print*,'readfits_r4e[1]: ',FILENAME(1:30)
          call readfits_r4e(FILENAME,pix,NAXIS1,NAXIS2,1)
          print*,'readfits_i4e[3]: ',FILENAME(1:30)
          call readfits_i4e(FILENAME,pxn,NAXIS1,NAXIS2,3)
         endif
 
      do i = 0001, NAXIS1
      do j = 0001, NAXIS2
         sat(i,j) = 0
         uuu(i,j) = 0
         if (.not.(pix(i,j).gt.-999)) then
             sat(i,j) = 2
             pix(i,j) = -999
             endif
         enddo
         enddo
 
      do i = 0001+2, NAXIS1-2
      do j = 0001+2, NAXIS2-2
         do ii = i-1, i+1
         do jj = j-1, j+1
            if (pxn(ii,jj).eq.0.and.sat(i,j).eq.0) sat(i,j) = 1
            enddo
            enddo
         enddo
         enddo
 
      call blank_edges(sat,NAXIS1,NAXIS2)
      call fill_pxf(pix,sat,pxf,pxg,NAXIS1,NAXIS2)
 
      do i = 0001, NAXIS1
      do j = 0001, NAXIS2
         if (sat(i,j).eq.2) pix(i,j) = -999.0
         enddo
         enddo
 
 
      CRPIX1 = r8_query_hdre(FILENAME,'CRPIX1  ',-1)
      CRPIX2 = r8_query_hdre(FILENAME,'CRPIX2  ',-1)
      CRVAL1 = r8_query_hdre(FILENAME,'CRVAL1  ',-1)
      CRVAL2 = r8_query_hdre(FILENAME,'CRVAL2  ',-1)
      CD1_1  = r8_query_hdre(FILENAME,'CD1_1   ',-1)
      CD1_2  = r8_query_hdre(FILENAME,'CD1_2   ',-1)
      CD2_1  = r8_query_hdre(FILENAME,'CD2_1   ',-1)
      CD2_2  = r8_query_hdre(FILENAME,'CD2_2   ',-1)
 
      print*,'CRPIX: ',CRPIX1,CRPIX2
      print*,'CRVAL: ',CRVAL1,CRVAL2
 
      if (.false.) then
         FILENAMEu = 'sat.fits'
         print*,'writfits_b1: ',FILENAME(1:30)
         call writfits_b1(FILENAMEu,sat,NAXIS1,NAXIS2)
 
         FILENAMEu = 'pxf.fits'
         print*,'writfits_r4: ',FILENAMEu(1:30)
         call writfits_r4(FILENAMEu,pxf,NAXIS1,NAXIS2)
 
         FILENAMEu = 'pxg.fits'
         print*,'writfits_r4: ',FILENAMEu(1:30)
         call writfits_r4(FILENAMEu,pxg,NAXIS1,NAXIS2)
 
         FILENAMEu = 'pix.fits'
         print*,'writfits_r4: ',FILENAMEu(1:30)
         call writfits_r4(FILENAMEu,pix,NAXIS1,NAXIS2)
         endif
 
      PROP = i4_query_hdre(FILENAME,'PROPOSID',-1)
      EXPT = r4_query_hdre(FILENAME,'EXPTIME ',-1)
      RDAT = rdate_header(FILENAME)
 
      call query_hdre(FILENAME,'FILTER  ',STREAM,-1)
 
      FILT = 'FXXXW'
      if (STREAM(1:1).eq.'F')  FILT = STREAM(1:5)
      if (STREAM(2:2).eq.'F')  FILT = STREAM(2:6)
 
 
      do k = 1, 4
         Ns = 0
         t_k(k) = -999
         DMAX = 0
         do i = 0001, NAXIS1, 5
         do j = 0001, NAXIS2, 5
            if (pix(i,j).gt.0) then
                dx = (i-NAXIS1/2)
                dy = (j-NAXIS2/2)
                tt = atan2(dy,dx)*180.0/3.14159
                kkx = 0
                do kk = 1, k-1
                   if (abs(tt-t_k(kk)-000).lt.45.or.
     .                 abs(tt-t_k(kk)-360).lt.45.or.
     .                 abs(tt-t_k(kk)+360).lt.45) kkx = kk
                   enddo
                dd = sqrt(dx**2+dy**2)
                Ns = Ns + 1
                if (dd.gt.DMAX.and.kkx.eq.0) then
                    DMAX = dd
                    i_k(k) = i
                    j_k(k) = j
                    t_k(k) = tt
                    endif
    3           continue
                endif
            enddo
            enddo
         print*,'  k: ',k,i_k(k),j_k(k),t_k(k),dmax,Ns
         BDRY_XR(k,1) = i_k(K)
         BDRY_YR(k,1) = j_k(K)
         enddo
 
      call cosort_rdd(t_k,BDRY_XR,BDRY_YR,4)
 
      do k = 1, 4
         print*,'k: ',k,t_k(K),BDRY_XR(k,1), BDRY_YR(k,1)
         enddo
 
      print*,'        '
      print*,'read_hstdrz_full: '
      print*,'        '
      print*,'CRPIX1: ',CRPIX1,CRPIX2
      print*,'CRVAL1: ',CRVAL1,CRVAL2
      print*,' CD1_1: ',CD1_1, CD1_2
      print*,' CD2_1: ',CD2_1, CD2_2
      print*,'LOFLAG: ',LOFLAG
      print*,'HIFLAG: ',HIFLAG
      print*,'BDRY_X: ',(BDRY_XR(k,1),k=1,4)
      print*,'BDRY_Y: ',(BDRY_YR(k,1),k=1,4)
      print*,'  FILT: ',FILT
      print*,'  EXPT: ',EXPT
      print*,'  RDAT: ',RDAT
      print*,'  PROP: ',PROP
      print*,'        '
 
      return
      end
 
      subroutine cosort_rdd(a,b,c,Ls)
      implicit none
 
      integer Ls
      real*4  a(Ls)
      real*8  b(Ls)
      real*8  c(Ls)
 
      real*8  aa,bb,cc
      integer L, LL
 
      do L = 1, Ls
         do LL = 1, Ls-1
            if (a(LL).gt.a(LL+1)) then
                aa = a(LL)
                bb = b(LL)
                cc = c(LL)
                a(LL) = a(LL+1)
                b(LL) = b(LL+1)
                c(LL) = c(LL+1)
                a(LL+1) = aa
                b(LL+1) = bb
                c(LL+1) = cc
                endif
            enddo
         enddo
 
      return
      end
 
 
 
c----------------------------------------------
c
c
      subroutine blank_edges(sat,PXDIMX,PXDIMY)
      implicit none
 
      integer PXDIMX
      integer PXDIMY
      byte sat(PXDIMX,PXDIMY)
 
      integer i, j
      integer iu,ju
      integer NIT
      integer NCHG
      integer SX, SY ! sense
 
      do i = 0001, PXDIMX
         sat(i, 0001 ) = 2
         sat(i,PXDIMY) = 2
         enddo
      do j = 0001, PXDIMY
         sat( 0001 ,j) = 2
         sat(PXDIMX,j) = 2
         enddo
 
      NIT = 0
    1 NIT = NIT + 1
      NCHG = 0
      do i = 0001+1, PXDIMX-1, 01
      do j = 0001+1, PXDIMX-1, 01
         do SX = 0, 1, 1
         do SY = 0, 1, 1
            iu = i
            ju = j
            if (SX.eq.1) iu = PXDIMX-(i-1)
            if (SY.eq.1) ju = PXDIMY-(j-1)
            if (sat(iu,ju).eq.1) then
               if (sat(iu+1,ju  ).eq.2.or.
     .             sat(iu-1,ju  ).eq.2.or.
     .             sat(iu  ,ju+1).eq.2.or.
     .             sat(iu  ,ju-1).eq.2.or.
     .             sat(iu+1,ju+1).eq.2.or.
     .             sat(iu+1,ju-1).eq.2.or.
     .             sat(iu-1,ju+1).eq.2.or.
     .             sat(iu-1,ju-1).eq.2) then
                   NCHG = NCHG + 1
                   sat(iu,ju) = 2
                   endif
               endif
            enddo
            enddo
         enddo
         enddo
      write(*,'(''blank_edges --- NIT: '',i4,1x,i8)') NIT,NCHG
      if (NCHG.gt.0) goto 1
 
      return
      end
 
 
 
 
      subroutine fill_pxf(pix,sat,pxf,pxg,PXDIMX,PXDIMY)
      implicit none
 
      integer PXDIMX, PXDIMY
      real pix(PXDIMX,PXDIMY)
      byte sat(PXDIMX,PXDIMY)
      real pxf(PXDIMX,PXDIMY)
      real pxg(PXDIMX,PXDIMY)
 
      integer i,  j
      integer ii, jj
      integer rr
      integer NS, NT
      integer L, Ls, Lo, LL
      integer il(99999)
      integer jl(99999)
      real    fl(99999), fmax
      real    pl(99999), pmax, ptot
 
      integer iu, ju
      integer sx, sy
      integer NIT, NCHG
      logical ISSAT
 
      do i = 0001, PXDIMX
      do j = 0001, PXDIMY
         pxf(i,j) = 0.
         if (sat(i,j).eq.1) then
            rr = 1
    1       rr = rr+1
            Ls = 0
            NT = 0
            NS = 0
            ptot = 0.
            do ii = -rr, rr
            do jj = -rr, rr
               if (ii**2+jj**2.le.(rr+0.5)**2) then
                   if (i+ii.ge.0001+1.and.i+ii.le.PXDIMX-1.and.
     .                 j+jj.ge.0001+1.and.j+jj.le.PXDIMY-1) then
                       NT = NT + 1
                       if (sat(i+ii,j+jj).eq.1) NS = NS + 1
                       Ls = Ls + 1
                       if (Ls.gt.99999) then
                           print*,'  i: ',i
                           print*,'  j: ',j
                           print*,' rr: ',rr
                           print*,' Ls: ',Ls
                           print*,' pt: ',ptot
                           stop 'Ls.gt.99999'
                           endif
                       ptot = ptot + max(pix(i+ii,j+jj),0.0)
                       endif
                   endif
               enddo
               enddo
            if (NS.gt.0.750*NT) goto 1
            pxf(i,j) = ptot
            endif
         enddo
         enddo
 
      do i = 0001, PXDIMX
      do j = 0001, PXDIMY
         pxg(i,j) = pix(i,j)
         enddo
         enddo
 
      if (.false.) then
      do i = 0001, PXDIMX
      do j = 0001, PXDIMY
         pxg(i,j) = pxf(i,j)
         enddo
         enddo
 
      NIT = 0
    2 NIT = NIT + 1
      NCHG = 0
      do i = 0001+1, PXDIMX-1, 01
      do j = 0001+1, PXDIMX-1, 01
         do SX = 0, 1, 1
         do SY = 0, 1, 1
            iu = i
            ju = j
            if (SX.eq.1) iu = PXDIMX-(i-1)
            if (SY.eq.1) ju = PXDIMY-(j-1)
            if (sat(iu,ju).eq.1) then
               if (pxg(iu,ju).lt.pxg(iu+1,ju  )) then
                   pxg(iu,ju) =  pxg(iu+1,ju  )
                   NCHG = NCHG + 1
                   endif
               if (pxg(iu,ju).lt.pxg(iu-1,ju  )) then
                   pxg(iu,ju) =  pxg(iu-1,ju  )
                   NCHG = NCHG + 1
                   endif
               if (pxg(iu,ju).lt.pxg(iu  ,ju+1)) then
                   pxg(iu,ju) =  pxg(iu  ,ju+1)
                   NCHG = NCHG + 1
                   endif
               if (pxg(iu,ju).lt.pxg(iu  ,ju-1)) then
                   pxg(iu,ju) =  pxg(iu  ,ju-1)
                   NCHG = NCHG + 1
                   endif
               endif
            enddo
            enddo
         enddo
         enddo
      write(*,'(''glom_pxf --- NIT: '',i4,1x,i8)') NIT,NCHG
      if (NCHG.gt.0) goto 2
      endif!
 
      Ns = 0
      do i = 0001, PXDIMX
      do j = 0001, PXDIMY
         if (sat(i,j).eq.1) then
             Ns = Ns + 1
             Ls = 1
             il(Ls) = i
             jl(Ls) = j
             pl(Ls) = pix(i,j)
   9         continue
             Lo = Ls
             do L = 1, Ls
                do ii = max(0001,il(L)-1), min(PXDIMX,il(L)+1)
                do jj = max(0001,jl(L)-1), min(PXDIMY,jl(L)+1)
                   if (sat(ii,jj).eq.1) then
                       do LL = 1, Ls
                          if (il(LL).eq.ii.and.jl(LL).eq.jj) goto 8
                          enddo
                       Ls = Ls + 1
                       if (Ls.gt.9999) then
                           print*,'  Ls: ',Ls
                           print*,'   i: ',i,j
                           print*,'  ii: ',ii,jj
                           stop
                           endif
                       il(Ls) = ii
                       jl(Ls) = jj
                       pl(Ls) = pix(ii,jj)
                       fl(Ls) = pxf(ii,jj)
                       endif
    8              continue
                   enddo
                   enddo
                enddo
             if (Ls.gt.Lo) goto 9
             ptot = 0
             pmax = 0
             fmax = 0
             ii   = 1
             jj   = 1
             do L = 1, Ls
                ptot = ptot + pl(L)
                if (pl(L).gt.pmax) then
                    pmax = pl(L)
                    endif
                if (fl(L).gt.fmax) then
                    fmax = fl(L)
                    ii = il(L)
                    jj = jl(L)
                    endif
                enddo
             do L = 1, Ls
                pix(il(L),jl(L)) = -999
                sat(il(L),jl(L)) = 3
                enddo
             pix(ii  ,jj  ) = ptot
             pix(ii-1,jj  ) = 0.
             pix(ii+1,jj  ) = 0.
             pix(ii  ,jj-1) = 0.
             pix(ii  ,jj+1) = 0.
             pix(ii-1,jj-1) = 0.
             pix(ii+1,jj-1) = 0.
             pix(ii-1,jj+1) = 0.
             pix(ii+1,jj+1) = 0.
             pix(ii-2,jj-2) = -999
             pix(ii-2,jj-1) = -999
             pix(ii-2,jj  ) = -999
             pix(ii-2,jj+1) = -999
             pix(ii-2,jj+2) = -999
             pix(ii+2,jj-2) = -999
             pix(ii+2,jj-1) = -999
             pix(ii+2,jj  ) = -999
             pix(ii+2,jj+1) = -999
             pix(ii+2,jj+2) = -999
             pix(ii+1,jj-2) = -999
             pix(ii  ,jj-2) = -999
             pix(ii-1,jj-2) = -999
             pix(ii-1,jj+2) = -999
             pix(ii  ,jj+2) = -999
             pix(ii+1,jj+2) = -999
             endif
         enddo
         enddo
 
      print*,'Ns: ',Ns
 
      return
      end
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/FITSIO/GEN/readfits_j2r.f"
c****                                         
c*********************************************
                                              
 
cc-----------------------------------------------------
c
c This reads in an image in jay's integer*2 format into
c a real*4 pixel array
c
      subroutine readfits_j2r(FILE,pix,NX,NY)
      implicit none
 
      character*(*) FILE
      integer NX, NY
      real*4 pix(NX,NY)
 
      character*70 INFO(10)
      common / fitsinfo / INFO
 
      integer naxes
      integer laxis(3)
 
      character*8  field
      character*20 stream
 
      real*4 pixu
      integer nbyte0
      integer nbyteE
      integer nbyte1
      integer nbyte2
      integer nbper
      integer i,ios, k
      integer ii, jj
      integer n
      integer NXU, NYU
 
      character*2880 buffc
      byte buffb(2880)
      equivalence (buffc,buffb)
 
      integer*2 ibuff(1440)
 
      integer ifirst, i1, i2
      integer j
 
      integer np1, np2, npt
      integer nextend
      integer nread
      real bscale, bzero
      integer bitpix
 
      integer r2i
 
      logical LINUX
      data LINUX/.true./
      logical DIAG
      data DIAG /.false./
 
 
      character*70 HDR(25)
      common/HDR/HDR
      character*199 FILEU
 
      FILEU = FILE
      do i = 75,2,-1
         if (FILE(i:i+4).eq.'.fits') FILEU = FILE(1:i+4)
         enddo
 
      open(10,file=FILEU,status='old',
     .     err=900,recl=2880,form='UNFORMATTED',
     .     access='DIRECT')
 
      if (DIAG) print*,'...opened'
 
      bscale = 1
      bzero  = 0
 
      naxes = -1
      laxis(1) = 1
      laxis(2) = 1
      laxis(3) = 1
      nextend = 0
 
      do i = 1, 10
         INFO(i) = ' '
         enddo
      do i = 1, 25
         HDR(i) = ' '
         enddo
      HDR(24) = '  1.000 '
 
      i = 0
      nread = 0
 100  continue
      i = i + 1
      read(10,rec=i,iostat=ios) buffc
      if (DIAG) print*,'READREC: ',i
      do k = 0, 35, 1
         if (DIAG) write(*,'(i4,1x,i4,1x,a80)')
     .                   i,k,buffc(k*80+1:k*80+80)
         field  = buffc(k*80+01:k*80+08)
         stream = buffc(k*80+11:k*80+31)
         if (field.eq.'NAXIS   ') read(stream,*) naxes
         if (field.eq.'NAXIS1  ') read(stream,*) laxis(1)
         if (field.eq.'NAXIS2  ') read(stream,*) laxis(2)
         if (field.eq.'NAXIS3  ') read(stream,*) laxis(3)
         if (field.eq.'NEXTEND ') read(stream,*) nextend
         if (field.eq.'BITPIX  ') read(stream,*) bitpix
         if (field.eq.'BSCALE  ') read(stream,*) bscale
         if (field.eq.'BZERO   ') read(stream,*) bzero
 
         if (field.eq.'EXPTIME ') INFO(01) = stream
         if (field.eq.'FILTNAM1') INFO(02) = stream
         if (field.eq.'FILENAME') INFO(03) = stream
         if (field.eq.'DATE-OBS') INFO(04) = stream
         if (field.eq.'TIME-OBS') INFO(05) = stream
         if (field.eq.'DEC_TARG') INFO(06) = stream
         if (field.eq.'RA_TARG ') INFO(07) = stream
         if (field.eq.'PA_V3   ') INFO(08) = stream
         if (field.eq.'PROPOSID') INFO(09) = stream
         if (field.eq.'CCDGAIN ') INFO(10) = stream
 
         if (field.eq.'CRPIX1  ') HDR(01) = stream
         if (field.eq.'CRPIX2  ') HDR(02) = stream
         if (field.eq.'CRVAL1  ') HDR(03) = stream
         if (field.eq.'CRVAL2  ') HDR(04) = stream
         if (field.eq.'CTYPE1  ') HDR(05) = stream
         if (field.eq.'CTYPE2  ') HDR(06) = stream
         if (field.eq.'CD1_1   ') HDR(07) = stream
         if (field.eq.'CD1_2   ') HDR(08) = stream
         if (field.eq.'CD2_1   ') HDR(09) = stream
         if (field.eq.'CD2_2   ') HDR(10) = stream
         if (field.eq.'ORIENTAT') HDR(11) = stream
         if (field.eq.'PA_APER ') HDR(12) = stream
         if (field.eq.'PA_V3   ') HDR(13) = stream
         if (field.eq.'DATE-OBS') HDR(14) = stream
         if (field.eq.'TIME-OBS') HDR(15) = stream
         if (field.eq.'EXPTIME ') HDR(16) = stream
         if (field.eq.'ROOTNAME') HDR(17) = stream
         if (field.eq.'TARGNAME') HDR(18) = stream
         if (field.eq.'RA_TARG ') HDR(19) = stream
         if (field.eq.'DEC_TARG') HDR(20) = stream
         if (field.eq.'PROPOSID') HDR(21) = stream
         if (field.eq.'FILTER  ') HDR(22) = stream
         if (field.eq.'FILTER  ') HDR(23) = stream
         if (field.eq.'FILTER1 ') HDR(22) = stream
         if (field.eq.'FILTER2 ') HDR(23) = stream
         if (field.eq.'VAFACTOR') HDR(24) = stream
 
         if (field.eq.'END     ') goto 101
         enddo
      goto 100
 101  continue
 
      nread = nread + 1
      if (DIAG) then
         print*,'----------------------------------------'
         print*,'  NREAD: ',nread
         print*,'NEXTEND: ',nextend
         print*,'  NAXIS: ',naxes
         print*,'  LAXIS: ',laxis(1),laxis(2),laxis(3)
         print*,' BITPIX: ',bitpix
         print*,' BSCALE: ',bscale
         print*,'  BZERO: ',bzero
         endif
 
      ifirst = i+1
      i1 = i
      i2 = i
 
      if (BITPIX.ne.16) then
         print*,'readfits_i2...: '
         print*,'                '
         print*,'   you called a routine to read in an'
         print*,'   unsigned i2 image, and the image you'
         print*,'   gave it has BITPIX = ',BITPIX
         print*,'   '
         print*,'   FILEU: ',FILEU
         print*,'   '
         stop
         endif
 
      nbper = 2*laxis(1)*laxis(2)
      npt   =   laxis(1)*laxis(2)
      nbyte1 = 1
      nbyte2 = nbper
      i1 = i+1 + nbyte1/2880
      i2 = i+1 + nbyte2/2880
 
      if (laxis(1).gt.NX.and.laxis(2).gt.NY) then
         print*,' not enough image space! '
         print*,' '
         print*,'  laxis1: ',laxis(1)
         print*,'      NX: ',NX
         print*,' '
         print*,'  laxis2: ',laxis(2)
         print*,'      NY: ',NY
         print*,' '
         stop
         endif
 
      NXU = laxis(1)
      NYU = laxis(2)
 
      if (DIAG) then
         print*,'     NX: ',NX
         print*,'     NY: ',NY
         print*,'  NBPER: ',nbper
         print*,'  NBYT1: ',nbyte1
         print*,'  NBYT2: ',nbyte2
         print*,' IFIRST: ',ifirst
         print*,'     I1: ',i1
         print*,'     I2: ',i2
         print*,'    NPT: ',NPT
         endif
      do i = i1, i2, 1
         read(10,rec=i,iostat=ios) buffc
         nbyte0 = (i-ifirst)*2880+   1
         nbyteE = (i-ifirst)*2880+2880
         np1 = (nbyte0-nbyte1)/2 + 1
         np2 = (nbyteE-nbyte1)/2 + 1
         np2 = min(np2,npt)
         call buff2pix_j2r(buffb,ibuff,0001,1440)
         do n = np1, np2, 1
            jj = n/NXU   + 1
            ii = n-NXU*(jj-1)
            pixu = ibuff(n-np1+1)*bscale+bzero
            if (pixu.gt.55000) pixu = 55000 + (pixu-55000)*5
            pix(ii,jj) = pixu
            enddo
         enddo
 
      if (DIAG) then
         print*,'  NBPER: ',nbper
         print*,'  NBYT1: ',nbyte1
         print*,'  NBYT2: ',nbyte2
         print*,' IFIRST: ',ifirst
         print*,'     I1: ',i1
         print*,'     I2: ',i2
         print*,'    NPT: ',NPT
         endif
 
 
      return
 
 900  continue
      print*,'READFITS_I2 ERROR'
      print*,'    FILE: ',FILE
      stop
 
      end
 
 
 
      subroutine buff2pix_j2r(buff,pix,n1,nt)
      implicit none
      byte buff(2880)
      integer*2 pix(*)
      integer n1,nt
 
      byte b(2)
      integer ii
      equivalence(ii,b)
 
      integer i, npu, nbu
 
      logical islinux
 
      do i = 1, 1440
         npu = n1+i-1
         nbu = (i-1)*2
         if (.not.islinux()) then
            b(1) = buff(nbu+1)
            b(2) = buff(nbu+2)
            endif
         if (islinux()) then
            b(2) = buff(nbu+1)
            b(1) = buff(nbu+2)
            endif
         if (npu.ge.1.and.npu.le.nt) pix(npu) = ii
         enddo
 
      return
      end
 
 
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/ROUTINES/STRING/str_contains.f"
c****                                         
c*********************************************
                                              
 
c-----------------------------------------
c
c this routine will return whether STRING1
c contains an incidence STRING2
c
       logical function str_contains(STRING1,L1,STRING2,L2)
       implicit none
 
       character*(*) STRING1
       integer L1
       character*(*) STRING2
       integer L2
 
       integer L1u
       integer L2u
 
       integer L1t
 
       L1u = L1
       L2u = L2
 
       if (L1u.eq.0) then
           L1u = 0
   11      continue
           if (STRING1(L1u+1:L1u+1).eq.' ') goto 12
           goto 11
   12      continue
           endif
 
       if (L2u.eq.0) then
           L2u = 0
   21      continue
           if (STRING1(L2u+1:L2u+1).eq.' ') goto 22
           goto 21
   22      continue
           endif
 
       str_contains = .true.
 
       do L1t = 1, L1u-L2u+1
          if (STRING1(L1t:L1t+L2u-1).eq.STRING2(1:L2u)) return
          enddo
 
       str_contains = .false.
       return
       end
 
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/ROUTINES/NOISE/noisesig.f"
c****                                         
c*********************************************
                                              
c
c this function will return random noise in terms of sigma.  The return value
c will be between -2 and 2 and will have a gaussian distribution centered about
c 1 with a sigma of 1  ( exp( (-x**2)/2 ) )
c
 
      real function noise_sig()
      implicit none
 
      real rand
 
      noise_sig = cos(3.1415927*2*rand(0))*sqrt(-2*log(rand(0)))
 
      return
      end
 
 
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/PSFs/PERT/smoo_psfpert_new.f"
c****                                         
c*********************************************
                                              
 
c-------------------------------------
c
c
      subroutine smoo_psfpert(mi,mo)
      implicit none
 
      real mi(101,101)
      real mo(101,101)
 
      real m0(101,101)
      real m3(101,101)
 
      real ma(101,101)
      real mb(101,101)
 
      real mx(101,101)
 
      integer i, j
      real    r
 
      real cenrat
      common /smoo_psf_/cenrat
 
 
      do i = 1, 101
      do j = 1, 101
         m0(i,j) = mi(i,j)
         enddo
         enddo
 
      call sm3plan(m0,m3,101,101)
 
      do i = 1, 101
      do j = 1, 101
         r = sqrt((i-51.)**2+(j-51.)**2)/4.0
         mx(i,j) = m0(i,j) - m3(i,j)
         enddo
         enddo
 
      call sm5quad(mx,ma,101,101)
      call sm7quad(mx,ma,101,101)
      call sm9quad(mx,ma,101,101)
      call sm5plan(mx,mb,101,101)
 
      do i = 1, 101
      do j = 1, 101
         r = sqrt((i-51.)**2+(j-51.)**2)/4.0
         mo(i,j) = m3(i,j) + mb(i,j)
         if (r.lt.08) mo(i,j) = m3(i,j) + (ma(i,j)+mb(i,j))/2
         if (r.lt.04) mo(i,j) = m3(i,j) +  ma(i,j)
         enddo
         enddo
 
 
 
      return
      end
 
 
 
 
c-----------------------------------
c
c
      subroutine sm3plan(r,s,NX,NY)
      implicit none
 
      integer NX, NY
      real r(NX,NY)
      real s(NX,NY)
 
      integer i, j
      integer im, jm
 
      real A
 
      do i = 1, NX
      do j = 1, NY
         A = 0.
         do im = max(i-1,1), min(i+1,NX)
         do jm = max(j-1,1), min(j+1,NX)
            A = A + r(im,jm)/09
            enddo
            enddo
         s(i,j) = A
         enddo
         enddo
 
      return
      end
 
c------------------------------------
c
c
      subroutine sm5plan(r,s,NX,NY)
      implicit none
 
      integer NX, NY
      real r(NX,NY)
      real s(NX,NY)
 
      integer i, j
      integer iu, ju
      integer im, jm
 
      real dx, dy
      real A, B, C
 
      real AA( 5, 5)
      data AA / ! SUM:     16.0
     .      0.2500,   0.5000,   0.5000,   0.5000,   0.2500,
     .      0.5000,   1.0000,   1.0000,   1.0000,   0.5000,
     .      0.5000,   1.0000,   1.0000,   1.0000,   0.5000,
     .      0.5000,   1.0000,   1.0000,   1.0000,   0.5000,
     .      0.2500,   0.5000,   0.5000,   0.5000,   0.2500/
 
      real BB( 5, 5)
      data BB / ! SUM:     0.0
     .     -1.0000,  -0.5000,   0.0000,   0.5000,   1.0000,
     .     -1.0000,  -0.5000,   0.0000,   0.5000,   1.0000,
     .     -1.0000,  -0.5000,   0.0000,   0.5000,   1.0000,
     .     -1.0000,  -0.5000,   0.0000,   0.5000,   1.0000,
     .     -1.0000,  -0.5000,   0.0000,   0.5000,   1.0000/
 
      real CC( 5, 5)
      data CC / ! SUM:     0.0
     .     -1.0000,  -1.0000,  -1.0000,  -1.0000,  -1.0000,
     .     -0.5000,  -0.5000,  -0.5000,  -0.5000,  -0.5000,
     .      0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
     .      0.5000,   0.5000,   0.5000,   0.5000,   0.5000,
     .      1.0000,   1.0000,   1.0000,   1.0000,   1.0000/
 
 
      do i = 1, NX
      do j = 1, NY
         iu = max(min(i,NX-2),3)
         ju = max(min(j,NY-2),3)
         A = 0.
         B = 0.
         C = 0.
         do im = 1, 5
         do jm = 1, 5
            A = A + AA(im,jm)*r(iu-3+im,ju-3+jm)/16
            B = B + BB(im,jm)*r(iu-3+im,ju-3+jm)/25
            C = C + CC(im,jm)*r(iu-3+im,ju-3+jm)/25
            enddo
            enddo
         dx = i-iu
         dy = j-ju
         s(i,j) = A + B*dx + C*dy
         enddo
         enddo
 
      return
      end
 
 
c------------------------------------
c
c
      subroutine sm5quad(r,s,NX,NY)
      implicit none
 
      integer NX, NY
      real r(NX,NY)
      real s(NX,NY)
 
      integer i, j
      integer iu, ju
      integer im, jm
 
      real dx, dy
      real A, B, C, D, E, F
 
      real AA( 5, 5)
      data AA / ! SUM:     1.0
     .     -1.8571,   0.2857,   1.0000,   0.2857,  -1.8571,
     .      0.2857,   2.4286,   3.1429,   2.4286,   0.2857,
     .      1.0000,   3.1429,   3.8571,   3.1429,   1.0000,
     .      0.2857,   2.4286,   3.1429,   2.4286,   0.2857,
     .     -1.8571,   0.2857,   1.0000,   0.2857,  -1.8571/
 
      real BB( 5, 5)
      data BB / ! SUM:     0.0
     .     -1.0000,  -0.5000,   0.0000,   0.5000,   1.0000,
     .     -1.0000,  -0.5000,   0.0000,   0.5000,   1.0000,
     .     -1.0000,  -0.5000,   0.0000,   0.5000,   1.0000,
     .     -1.0000,  -0.5000,   0.0000,   0.5000,   1.0000,
     .     -1.0000,  -0.5000,   0.0000,   0.5000,   1.0000/
 
      real CC( 5, 5)
      data CC / ! SUM:     0.0
     .     -1.0000,  -1.0000,  -1.0000,  -1.0000,  -1.0000,
     .     -0.5000,  -0.5000,  -0.5000,  -0.5000,  -0.5000,
     .      0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
     .      0.5000,   0.5000,   0.5000,   0.5000,   0.5000,
     .      1.0000,   1.0000,   1.0000,   1.0000,   1.0000/
 
      real DD( 5, 5)
      data DD / ! SUM:     0.0
     .      0.7143,  -0.3571,  -0.7143,  -0.3571,   0.7143,
     .      0.7143,  -0.3571,  -0.7143,  -0.3571,   0.7143,
     .      0.7143,  -0.3571,  -0.7143,  -0.3571,   0.7143,
     .      0.7143,  -0.3571,  -0.7143,  -0.3571,   0.7143,
     .      0.7143,  -0.3571,  -0.7143,  -0.3571,   0.7143/
 
      real EE( 5, 5)
      data EE / ! SUM:     0.0
     .      1.0000,   0.5000,   0.0000,  -0.5000,  -1.0000,
     .      0.5000,   0.2500,   0.0000,  -0.2500,  -0.5000,
     .      0.0000,   0.0000,   0.0000,   0.0000,   0.0000,
     .     -0.5000,  -0.2500,   0.0000,   0.2500,   0.5000,
     .     -1.0000,  -0.5000,   0.0000,   0.5000,   1.0000/
 
      real FF( 5, 5)
      data FF / ! SUM:     0.0
     .      0.7143,   0.7143,   0.7143,   0.7143,   0.7143,
     .     -0.3571,  -0.3571,  -0.3571,  -0.3571,  -0.3571,
     .     -0.7143,  -0.7143,  -0.7143,  -0.7143,  -0.7143,
     .     -0.3571,  -0.3571,  -0.3571,  -0.3571,  -0.3571,
     .      0.7143,   0.7143,   0.7143,   0.7143,   0.7143/
 
 
      do i = 1, NX
      do j = 1, NY
         iu = max(min(i,NX-2),3)
         ju = max(min(j,NY-2),3)
         A = 0.
         B = 0.
         C = 0.
         D = 0.
         E = 0.
         F = 0.
         do im = 1, 5
         do jm = 1, 5
            A = A + AA(im,jm)*r(iu-3+im,ju-3+jm)/25
            B = B + BB(im,jm)*r(iu-3+im,ju-3+jm)/25
            C = C + CC(im,jm)*r(iu-3+im,ju-3+jm)/25
            D = D + DD(im,jm)*r(iu-3+im,ju-3+jm)/25
            E = E + EE(im,jm)*r(iu-3+im,ju-3+jm)/25
            F = F + FF(im,jm)*r(iu-3+im,ju-3+jm)/25
            enddo
            enddo
         dx = i-iu
         dy = j-ju
         s(i,j) = A +
     .            B*dx    + C*dy    +
     .            D*dx**2 + E*dx*dy + F*dy**2
         enddo
         enddo
 
      return
      end
 
 
c------------------------------------
c
c
      subroutine sm7quad(r,s,NX,NY)
      implicit none
 
      integer NX, NY
      real r(NX,NY)
      real s(NX,NY)
 
      integer i, j
      integer iu, ju
      integer im, jm
 
      real*8 A
      real AA(7,7)
 
      data AA / ! SUM:     49.0
     .-2.3333,-0.6667, 0.3333, 0.6667, 0.3333,-0.6667,-2.3333,
     .-0.6667, 1.0000, 2.0000, 2.3333, 2.0000, 1.0000,-0.6667,
     . 0.3333, 2.0000, 3.0000, 3.3333, 3.0000, 2.0000, 0.3333,
     . 0.6667, 2.3333, 3.3333, 3.6667, 3.3333, 2.3333, 0.6667,
     . 0.3333, 2.0000, 3.0000, 3.3333, 3.0000, 2.0000, 0.3333,
     .-0.6667, 1.0000, 2.0000, 2.3333, 2.0000, 1.0000,-0.6667,
     .-2.3333,-0.6667, 0.3333, 0.6667, 0.3333,-0.6667,-2.3333/
 
      do i = 1, NX
      do j = 1, NY
         A = 0.
         do im = 1, 7
         do jm = 1, 7
            iu = i + (im-4)
            ju = j + (jm-4)
            if (iu.ge.001.and.iu.le.NX.and.
     .          ju.ge.001.and.ju.le.NY) then
                A = A + AA(im,jm)*r(iu,ju)/49
                endif
            enddo
            enddo
         s(i,j) = A
         enddo
         enddo
 
      return
      end
 
 
 
c------------------------------------
c
c
      subroutine sm9quad(r,s,NX,NY)
      implicit none
 
      integer NX, NY
      real r(NX,NY)
      real s(NX,NY)
 
      integer i, j
      integer iu, ju
      integer im, jm
 
      real*8 A
 
      real AA( 9, 9)
      data AA / ! SUM:     1.0
     .      0.0000,  -1.9501,  -0.7828,  -0.0824,   0.1511,
     .               -0.0824,  -0.7828,  -1.9501,   0.0000,
     .     -1.9501,  -0.3159,   0.8514,   1.5518,   1.7853,
     .                1.5518,   0.8514,  -0.3159,  -1.9501,
     .     -0.7828,   0.8514,   2.0187,   2.7191,   2.9526,
     .                2.7191,   2.0187,   0.8514,  -0.7828,
     .     -0.0824,   1.5518,   2.7191,   3.4195,   3.6529,
     .                3.4195,   2.7191,   1.5518,  -0.0824,
     .      0.1511,   1.7853,   2.9526,   3.6529,   3.8864,
     .                3.6529,   2.9526,   1.7853,   0.1511,
     .     -0.0824,   1.5518,   2.7191,   3.4195,   3.6529,
     .                3.4195,   2.7191,   1.5518,  -0.0824,
     .     -0.7828,   0.8514,   2.0187,   2.7191,   2.9526,
     .                2.7191,   2.0187,   0.8514,  -0.7828,
     .     -1.9501,  -0.3159,   0.8514,   1.5518,   1.7853,
     .                1.5518,   0.8514,  -0.3159,  -1.9501,
     .      0.0000,  -1.9501,  -0.7828,  -0.0824,   0.1511,
     .               -0.0824,  -0.7828,  -1.9501,   0.0000/
 
      do i = 1, NX
      do j = 1, NY
         A = 0.
         do im = 1, 9
         do jm = 1, 9
            iu = i + (im-5)
            ju = j + (jm-5)
            if (iu.ge.001.and.iu.le.NX.and.
     .          ju.ge.001.and.ju.le.NY) then
                A = A + AA(im,jm)*r(iu,ju)/77
                endif
            enddo
            enddo
         s(i,j) = A
         enddo
         enddo
 
      return
      end
 
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/ROUTINES/SATN/fitsat_ccdXthenY.f"
c****                                         
c*********************************************
                                              
 
c------------------------------------------------------------
c
c
c
c
      subroutine fitsat_ccdXthenY(psfu,pixu,NAXIS1,NAXIS2,SKYBAR,
     .                            xfit,yfit,zfit,ysigfit)
      implicit none
 
      real    psfu(101,101)
      integer NAXIS1, NAXIS2
      real    pixu(NAXIS1,NAXIS2)
      real    SKYBAR
      real    xfit, yfit, zfit, ysigfit
 
      integer Ls, L
      integer il(9999)
      integer jl(9999)
      real    pl(9999)
      real    fl(9999), fu(9999), fv(9999)
      integer gl(9999), gtot
 
      real    xl(9999)
 
      integer i0, j0
      integer ie, je
      real    dx, dy
 
      integer  i,  j
      real     rpsf_phot
 
      real    xmin, zmin, zminx
      real    emin, e
      integer idx,  idy,  izz
      real    x, y, z, findz
 
      integer NIT, id
      real    ftot, ptot
      real    ebar, emax, etot, sig
      integer lmax
      integer NREJ
 
      real*8  p_ii(11)
      real*8  f_ii(11)
      integer j_ii(11)
      integer ii, iiu
      integer jj, jju
      real    xpsf, ypsf
      real    dxu
      real    ss, mbar_sky
 
      real    p_imin(5)
      real    f_imin(5)
      integer j_imin(5)
 
      integer juse
 
      integer jmin, j1, j2
      real    e_j(6001)
 
      integer Uu, Us
      real    xu(999), yu(999)
      real    wu(999), wtot
      integer gu(999)
      real    eu(999)
      real    su(999)
      real    ybar
      real    ysig
 
      character cchar
 
      i0 = int(xfit+0.5)
      j0 = int(yfit+0.5)
 
 
      if (.false.) then
         print*,' '
         print*,'ENTER fitsat_ccdXthenY...'
         print*,' '
         print*,'   i0: ',i0,xfit
         print*,'   j0: ',j0,yfit
         print*,' '
         write(*,138) (i,i=i0-10,i0+10)
         do j = j0+10, j0-10,-1
            write(*,139) j,
     .          (min(int(pixu(i,j)+0.5),99999),i=i0-10,i0+10)
            enddo
  139    format(1x,i4.4,1x,10i6,3x,i5,3x,10i6)
  138    format(1x,  4x,1x,10i6,3x,i5,3x,10i6)
         print*,' '
         print*,' '
         endif
 
 
c------------------------
c
c integrate up the pixels
c
      do ii = -5, 5
         jju = 2
   3     if (pixu(i0+ii,j0+jju).gt.55000.or.
     .       pixu(i0+ii,j0-jju).gt.55000) then
             jju = jju + 1
             if (jju.lt.25) goto 3
             endif
         p_ii(6+ii) = 0.
         do jj = -jju, jju
            p_ii(6+ii) = p_ii(6+ii) + (pixu(i0+ii,j0+jj)-ss)
            enddo
         j_ii(6+ii) = jju
c        write(*,'(i4,1x,f14.1,1x,i4)') i,p_ii(6+ii),j_ii(6+ii)
         enddo
 
c----------------------
c
c find the best x shift
c
      emin = 9e9
      xmin = 0.00
      zmin = 0.00
      do idx = -2000, 2000, 1
         dx = idx*0.001
         do ii = -5, 5
            f_ii(6+ii) = 0.
            do jj = -j_ii(6+ii), j_ii(6+ii)
               xpsf = ii - dx
               ypsf = jj
               f_ii(6+ii) = f_ii(6+ii) + rpsf_phot(xpsf,ypsf,psfu)
               enddo
            enddo
         ptot = 0.
         ftot = 0.
         do ii = -2, 2
            ptot = ptot + p_ii(6+ii)
            ftot = ftot + f_ii(6+ii)
            enddo
         z = ptot/ftot
         etot = 0.
         do ii = -2, 2
            e    = abs(p_ii(6+ii)-z*f_ii(6+ii))
            sig  = sqrt(p_ii(6+ii)+0.01**2*p_ii(6+ii)**2)
            etot = etot + (e/sig)**2
            enddo
         etot = sqrt(etot/4)
c        write(87,187) idx, etot,z,ptot,ftot
c 187    format(1x,i5,1x,f12.5,1x,f12.0,1x,f12.0,1x,f8.6)
         if (etot.lt.emin) then
             emin = etot
             xmin = idx*0.001
             zmin = z
             do ii = -2, 2
                p_imin(3+ii) = p_ii(6+ii)
                f_imin(3+ii) = f_ii(6+ii)
                j_imin(3+ii) = j_ii(6+ii)
                enddo
             endif
         enddo
 
c     write(88,188) emin,xmin,zmin
c 188 format(f12.2,1x,f8.3,1x,f12.1)
 
      zminx = zmin
 
c     print*,'OUTPUT FORT89...'
c     do ii = -2, 2
c        write( *,189) p_imin(3+ii), f_imin(3+ii), zmin,
c    .                 j_imin(3+ii)
c        write(89,189) p_imin(3+ii), f_imin(3+ii), zmin,
c    .                 j_imin(3+ii)
c        enddo
c 189 format(1x,f9.1,1x,f8.5,1x,f9.0,1x,i2)
 
      wtot = 0.
      Us   = 0
      do ii = -8, 8
         juse = max(2,abs(ii))
         Ls = 0
         do jj = -juse, juse
            Ls = Ls + 1
            xl(Ls) = ii - xmin
            jl(Ls) = jj
            pl(Ls) = pixu(i0+ii,j0+jj)-ss
            if (pixu(i0+ii,j0+jj).gt.60000) then
                Ls = 0
                goto 3333
                endif
            enddo
c        do L = 1, Ls
c           write(91,191) ii, L, xl(L), jl(L), pl(L)
c 191       format(1x,i3,1x,i4.4,1x,f8.3,1x,i4,1x,f9.1)
c           enddo
         do idy = -3000, 3000
            ptot = 0.
            ftot = 0.
            do L = 1, Ls
               xpsf = xl(L)
               ypsf = jl(L) - idy*0.001
               fl(L) = rpsf_phot(xpsf,ypsf,psfu)
               ptot = ptot + pl(L)
               ftot = ftot + fl(L)
               enddo
            z = ptot/ftot
            if (z.lt.0) goto 3332
            etot = 0.
            do L = 1, Ls
               e = abs(pl(L)-z*fl(L))
               sig  = sqrt(abs(pl(L))+0.01**2*pl(L)**2)
               etot = etot + (e/sig)**2
               enddo
            etot = sqrt(etot/(Ls-1))
            e_j(3001+idy) = etot
c           write(92,192) ii,idy,etot,z
c 192       format(1x,i3,1x,i5,1x,f12.3,1x,f15.1)
 3332       continue
            enddo
         emin = 9e9
         jmin = 0
         do j = 1, 6001
            if (e_j(j).lt.emin) then
                emin = e_j(j)
                jmin = j
                endif
            enddo
         do j1 = jmin, 0001,-1
            if (e_j(j1).gt.emin*1.5) goto 3301
            enddo
 3301    continue
         do j2 = jmin, 6001,+1
            if (e_j(j2).gt.emin*1.5) goto 3302
            enddo
 3302    continue
c        write(*,1333) ii,Ls,emin,(jmin-3001),j2-j1
         Us = Us + 1
         xu(Us) = xpsf
         yu(Us) = (jmin-3001)*0.001
         eu(Us) = 0.001*(j2-j1)/2.0
         su(Us) = eu(Us)
         wu(Us) = 1/eu(Us)
         gu(Us) = 1
         wtot = wtot + wu(Us)
 1333    format(1x,i4,1x,i4,1x,f10.2,1x,i4,1x,i4)
 3333    continue
         enddo
 
c     print*,' '
c     do Uu = 1, Us
c        write(*,1334) ii,Uu,xu(Uu),yu(Uu),wu(Uu),eu(Uu),su(Uu)
c1334    format(1x,i3,1x,i4,1x,f10.4,1x,f10.4,8(1x,f8.4))
c        enddo
 
      call rbarsigw_empir(yu,wu,gu,eu,su,Us,ybar,ysig,Uu)
 
c     print*,' '
c     write(*,'(''  ybar: '',f8.4)') ybar
c     write(*,'(''  ysig: '',f8.4)') ysig
c     print*,' '
c1338 format(1x,3x,1x,4x,1x,10x,1x,f10.4,1x,f8.4)
 
 
c      print*,' '
c      do Uu = 1, Us
c         write(*,1339) ii,Uu,xu(Uu),yu(Uu),
c     .                       wu(Uu),eu(Uu),su(Uu),gu(Uu)
c 1339    format(1x,i3,1x,i4,1x,f10.4,1x,f10.4,
c     .                1x,f8.4,3x,f8.4,3x,f8.4,3x,i1)
c         enddo
c      print*,' '
 
c     if (.not.(xmin.lt.9.99)) xmin = 9.99
c     if (.not.(emin.lt.9999)) emin = 9999
c     if (.not.(zmin.lt.9e11)) zmin = 9e11
c     write(98,198) xmin, emin,zmin,pixu(i0,j0),
c    .              ybar, ysig
c 198 format(1x,f7.3,1x,f12.5,1x,f12.0,1x,f8.1,
c    .       3x,f7.3,1x,f7.3)
 
      xfit = i0 + xmin
      yfit = j0 + ybar
      zfit = zmin
 
      write(99,199) xfit, yfit, zfit, emin,
     .              pixu(i0,j0),
     .              xmin, ybar, ysig
  199 format(1x,f9.3,1x,f9.3,1x,f11.1,1x,f7.2,
     .               1x,f9.1,1x,
     .               1x,f8.3,1x,f8.3,1x,f8.3)
 
      ysigfit = ysig
 
c     close(87)
c     close(91)
c     close(92)
c     close(89)
c     close(98)
c     print*,'CONTINUE? '
c     read(*,*) cchar
 
      return
      end
 
 
      real function findz(pl,fl,Ls,e,zi)
      implicit none
 
      integer Ls
      real pl(Ls)
      real fl(Ls)
      real e
      real zi
 
      integer L
      real z0, zP, zM
      real e0, eP, eM
 
      e0 = 0.
      eP = 0.
      eM = 0.
      z0 = zi
 
    1 continue
      zP = z0*1.005
      zM = z0*0.995
      do L = 1, Ls
         e0 = e0 + abs(fl(L)*z0-pl(L))
         eP = eP + abs(fl(L)*zP-pl(L))
         eM = eM + abs(fl(L)*zM-pl(L))
         enddo
      if (eP.lt.e0) then
          z0 = zP
          goto 1
          endif
      if (eM.lt.e0) then
          z0 = zM
          goto 1
          endif
      e = e0
      findz = z0
 
      return
      end
 
 
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/ROUTINES/SATN/rbarsigw_empir.f"
c****                                         
c*********************************************
                                              
 
c-------------------------------------------------------
c
c
c
      subroutine rbarsigw_empir(yu,wu,gu,eu,su,Us,ybar,ysig,Uu)
      implicit none
 
      integer Us
      real    yu(Us)
      real    wu(Us)
      integer gu(Us)
      real    eu(Us)
      real    su(Us)
      real    ybar
      real    ysig
      integer Uu
 
      integer U
      real    ytot
      real    wtot
      real    ztot
      integer ntot
      real    stot
 
      real    fac
 
      ytot = 0.
      wtot = 0.
      ntot = 0
      do U = 1, Us
         ytot = ytot + gu(U)*wu(U)*yu(U)
         wtot = wtot + gu(U)*wu(U)
         ntot = ntot + gu(U)
         enddo
      ybar = ytot/wtot
 
      ytot = 0.
      wtot = 0.
      ntot = 0
      do U = 1, Us
         ytot = ytot + gu(U)*wu(U)*(yu(U)-ybar)**2
         wtot = wtot + gu(U)*wu(U)
         ntot = ntot + gu(U)
         enddo
      ysig = sqrt(ytot/wtot)
 
      ytot = 0.
      ntot = 0
      do U = 1, Us
         ytot = ytot + gu(U)*(yu(U)-ybar)**2/eu(U)**2
         ntot = ntot + gu(U)
         enddo
 
      fac = (ntot-1)/ytot
 
c     print*,' '
c     print*,' '
c     print*,' '
c     print*,' '
c
c     print*,' '
c     print*,'---> fac: ',fac
c     print*,'     mod: ',sqrt(fac)
c     print*,'    ytot: ',ytot
c     print*,' '
 
      do U = 1, Us
         su(U) = eu(U)/sqrt(fac)
         enddo
 
      ytot = 0.
      ztot = 0.
      ntot = 0.
c     write(*,108)
c     write(*,109)
      do U = 1, Us
c        write(*,119) U, yu(U), (yu(U)-ybar),wu(U)/wtot,
c    .                   eu(U), (yu(U)-ybar)**2/eu(U)**2,
c    .                     sqrt((yu(U)-ybar)**2/eu(U)**2),
c    .                   su(U), (yu(U)-ybar)**2/su(U)**2,
c    .                     sqrt((yu(U)-ybar)**2/su(U)**2),
c    .                   gu(U)
         ytot = ytot + gu(U)*(yu(U)-ybar)**2/eu(U)**2
         ztot = ztot + gu(U)*(yu(U)-ybar)**2/su(U)**2
         ntot = ntot + gu(U)
  108    format(1x,'   U',1x,
     .          1x,'...yu...',
     .          1x,'...dyu..',
     .          1x,'...wu...',
     .          1x,'...eu...',
     .          1x,'..erat2.',
     .          1x,'..erat..',
     .          1x,'...su...',
     .          1x,'..srat2.',
     .          1x,'..srat..')
  109    format(1x,'....',1x,
     .          1x,'........',
     .          1x,'........',
     .          1x,'........',
     .          1x,'........',
     .          1x,'........',
     .          1x,'........',
     .          1x,'........',
     .          1x,'........',
     .          1x,'........')
  119    format(1x,i4,1x,9(1x,f8.3),1x,i1)
  118    format(1x,i4,1x,9x,9x,9x,9x,1x,f8.3,9x,9x,1x,f8.3)
         enddo
c     print*,' '
c     write(*,118) ntot,ytot, ztot
c     print*,' '
 
      stot = 0.
      do U = 1, Us
         stot = stot + 1/su(U)**2
         enddo
 
c     print*,'       '
c     write(*,'(1x,i4,1x,f9.3)') ntot,sqrt(1/stot)
c     print*,'       '
 
      ysig = sqrt(1/stot)
 
      if (.not.(ybar.gt.-9.99)) ybar =  9.99
      if (.not.(ysig.gt. 0.00)) ysig =  9.99
 
      return
      end
 
 
 
                                              
c*********************************************
c****                                         
c**** #include "/user/jayander/FORTRAN/PARLOR/islinux.f"
c****                                         
c*********************************************
                                              
 
c      program q
c      implicit none
c
c      logical islinux, islinux_var
c
c      islinux_var = islinux()
c
c      islinux_var = islinux()
c
c      stop
c      end
 
 
      logical function islinux()
      implicit none
 
      logical done
      logical islinux_save
 
      integer i
      data i/0/
      data islinux_save/.true./
 
      common /islinux_/i,islinux_save
 
      byte    b(2)
      equivalence(i,b)
 
      if (i.eq.1) goto 1
 
      i = 1
      islinux_save = .false.
      if (b(1).eq.1) islinux_save = .true.
 
    1 islinux = islinux_save
      return
      end
 
 
 
