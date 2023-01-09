\ProvidesClass{thesis}[2020/02/02 IntelliSec thesis template]

\LoadClass[%
  a4paper,%
  11pt,% <10pt, 9pt>
  rgb, %<cmyk>
  oneside, %
  %style=screen, %
  % mono,
  smallchapters,
  extramargin,
  numbers=noendperiod
]{scrartcl}


%\KOMAoption{chapterprefix}{false}

\pdfsuppresswarningpagegroup=1
\usepackage[utf8x]{inputenc}

\RequirePackage{babel}
\RequirePackage{booktabs}
\RequirePackage{breakurl}
\usepackage{ifthen}
\RequirePackage[hyphens]{url} %needs to be before hyperref
\RequirePackage[breaklinks,hidelinks,linktoc=all]{hyperref}
\RequirePackage{kvoptions}
\SetupKeyvalOptions{family=thesis,prefix=thesis@,setkeys=\setkeys}
\RequirePackage{microtype}
\RequirePackage[numbers,square,sort&compress]{natbib}
\RequirePackage[binary-units=true]{siunitx}
\sisetup{%
 group-separator={,},%
 group-minimum-digits=4,%
 group-digits=integer,%
 detect-weight=true,%
 detect-family=true%
}
\RequirePackage{tabularx}
\RequirePackage{xcolor}
\RequirePackage{xspace}


\DeclareStringOption{name}[Steff Student]
\DeclareStringOption{id}[1337]
\DeclareStringOption{mail}[steff@student.kit.edu]
\DeclareStringOption{titlede}[Mein Titel]
\DeclareStringOption{titleen}[My Title]
\DeclareStringOption{thesis}[BSc]
\DeclareStringOption{registration}[1. February 2020]
\DeclareStringOption{submission}[31. July]
\DeclareStringOption{degree}[Informatik]
\DeclareStringOption{supervisor}[Jun.-Prof. Dr. Christian Wressnegger]
\DeclareStringOption{cosupervisor}[Prof. Dr. N.N]
\DeclareBoolOption[false]{expose}
\DeclareBoolOption[false]{de}
\ProcessKeyvalOptions*\relax

\newcommand*{\thesissetup}[1]{
  \kvsetkeys{thesis}{#1}
}

% Sorry! I have no clue how to do this properly :/
\newcommand{\myname}{\thesis@name}
\newcommand{\myid}{\thesis@id}
\newcommand{\mymail}{\thesis@mail}
\newcommand{\mytitlede}{\thesis@titlede}
\newcommand{\mytitleen}{\thesis@titleen}
\newcommand{\mythesistype}{\thesis@thesis}
\newcommand{\myregistrationdate}{\thesis@registration}
\newcommand{\mysubmissiondate}{\thesis@submission}
\newcommand{\mydegree}{\thesis@degree}
\newcommand{\mysupervisor}{\thesis@supervisor}
\newcommand{\mycosupervisor}{\thesis@cosupervisor}
\newcommand{\myexpose}{\boolean{thesis@expose}}
\newif\ifisexpose\isexposefalse
\ifthenelse{\boolean{thesis@expose}}{\isexposetrue}{}

\ifthenelse{\boolean{thesis@de}}{
  \newcommand{\thesistype}{%
    \ifthenelse{\equal{\mythesistype}{BSc}}
    {Bachelorarbeit}
    {Masterarbeit}\xspace
  }
  \newcommand{\kit}{Karlsruher Institut für Technologie~(KIT)\xspace}
  \selectlanguage{ngerman}
  \newcommand{\mytitle}{\mytitlede}
 }{
  \newcommand{\thesistype}{%
    \ifthenelse{\equal{\mythesistype}{BSc}}
    {Bachelor's thesis}
    {Master's thesis}\xspace
  }
  \newcommand{\kit}{Karlsruhe Institute of Technology~(KIT)\xspace}
  \newcommand{\mytitle}{\mytitleen}
}


\newcommand{\fig}[1]{\hyperref[#1]{\autoref{#1}}}
\newcommand{\figs}[1]{\hyperref[#1]{\autoref{#1}}}
\newcommand{\tab}[1]{\hyperref[#1]{\autoref{#1}}}
\newcommand{\tabs}[1]{\hyperref[#1]{\autoref{#1}}}
\newcommand{\sect}[1]{\hyperref[#1]{\autoref{#1}}}
\newcommand{\sects}[1]{\hyperref[#1]{\autoref{#1}}}

\definecolor{kitgreen}{HTML}{009682}



\makeatletter
\renewcommand\paragraph{\@startsection{paragraph}{4}{\z@}%
                                   {2ex \@plus 1ex \@minus .2ex}%
                                   {-1em}%
                                   {\normalsize\bfseries}}
\addtokomafont{section}{\color{kitgreen}}
\addtokomafont{subsection}{\color{kitgreen}}
\addtokomafont{subsubsection}{\color{kitgreen}}
\makeatother

\renewcommand{\rmdefault}{\sfdefault}




\makeatletter
\hypersetup{
  pdftitle = \thesis@titleen,
  pdfauthor = \thesis@name,
  pdfsubject = {},
  pdfkeywords = {}
}
\makeatother


\renewcommand{\maketitle}{%
\thispagestyle{empty}
\begin{center}
  \ifisexpose
    \Large{
      \ifthenelse{\boolean{thesis@de}}{Exposé zur}{Exposé of the}
      \thesistype
    }\\
  \fi
  \LARGE
  {\color{kitgreen}\textbf{\mytitle}}\\[10mm]
  \normalsize \myname\\\kit
\end{center}

\vfill

\begin{table}[h]
  \centering
  \begin{tabularx}{.7\textwidth}{rX}
    \hline
    \textbf{\ifthenelse{\boolean{thesis@de}}{Deutscher Titel }{German Title     }} & \mytitlede                    \\\hline
    \textbf{\ifthenelse{\boolean{thesis@de}}{Englischer Titel}{English Title    }} & \mytitleen                    \\\hline
    \textbf{\ifthenelse{\boolean{thesis@de}}{Anmeldedatum    }{Registration Date}} & \myregistrationdate           \\\hline
    \textbf{\ifthenelse{\boolean{thesis@de}}{Abgabedatum     }{Submission Date  }} & \mysubmissiondate             \\\hline
    \textbf{\ifthenelse{\boolean{thesis@de}}{Erstprüfer      }{Supervisor       }} & \mysupervisor                 \\\hline
    \textbf{\ifthenelse{\boolean{thesis@de}}{Zweitprüfer     }{Co-Examiner      }} & \mycosupervisor               \\\hline
    \textbf{\ifthenelse{\boolean{thesis@de}}{Matrikelnummer  }{Student ID       }} & \myid                         \\\hline
    \textbf{\ifthenelse{\boolean{thesis@de}}{E-Mail          }{E-Mail           }} & \href{mailto:\mymail}{\mymail}\\\hline
    \textbf{\ifthenelse{\boolean{thesis@de}}{Studiengang     }{Degree           }} & \mydegree                     \\\hline
  \end{tabularx}
\end{table}
\bigskip
\begin{center}
  \today
\end{center}
\clearpage
}
