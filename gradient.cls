%
% LaTeX Class based on a design by F. Dornhecker
%

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{gradient}
 
% Based on the memoir class
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{memoir}}
\ExecuteOptions{openany}
\ProcessOptions\relax
\LoadClass{memoir}

\RequirePackage[table]{xcolor}
\RequirePackage{afterpage}
\RequirePackage{hphecker}
\RequirePackage{caption}
\RequirePackage{graphicx}
\RequirePackage{charter}
\RequirePackage[bitstream-charter]{mathdesign}


% fix Undefined color `BLACK'
\colorlet{BLACK}{black}

% prevent indents for new paragraphs
\setlength{\parindent}{0em}


% blank page command. usage: \afterpage{\blankpage}
\newcommand\blankpage{%
  \null
  \thispagestyle{empty}
  \newpage}


% chapter style
\newcommand{\chaptercolor}{black}
\makechapterstyle{box}{
  \renewcommand*{\printchaptername}{}
  \renewcommand*{\chapnumfont}{\normalfont\sffamily\huge\bfseries}
  \renewcommand*{\printchapternum}{
		\pagecolor{\chaptercolor}\afterpage{\nopagecolor}
		{\fontsize{34pt}{10pt}\bfseries\color{white} Chapter \thechapter}
  }
  \renewcommand*{\chaptitlefont}{\fontsize{34pt}{10pt}\itshape\color{white}}
  \renewcommand*{\printchaptertitle}[1]{\chaptitlefont##1}
}
\chapterstyle{box}

% bib/toc headers
\renewcommand\printtoctitle[1]{{\fontsize{34pt}{10pt}\bfseries #1}}
\renewcommand\bibname{{\color{black} Bibliography}}


% epigraph style
\newcommand{\epiwidth}{.8\linewidth}
\newcommand{\epirulethickness}{0.5pt}
\renewcommand{\epigraph}[2]{
  \vfill{\color{white}
    \rule{\epiwidth}{\epirulethickness}\\\\
    \begin{minipage}{\epiwidth} #1 \end{minipage}
    #2
  \vspace{1cm}
  }\newpage\thispagestyle{rulednotitle}}


% section style
\setsecheadstyle{\LARGE\bfseries\raggedright}
\setbeforesecskip{-4.0ex plus -.5ex minus -.2ex}
\setaftersecskip{1.3ex plus .2ex}

\setsecnumdepth{subsection} % turn on subsec numbers
\setsubsecheadstyle{\large\bfseries\itshape\raggedright}
\setbeforesubsecskip{-3.0ex plus -.5ex minus -.2ex}
\setaftersubsecskip{1ex plus .2ex}
\setsubsubsecheadstyle{\normalfont\bfseries\raggedright}


% quote style
\let\oldquote\quote
\let\endoldquote\endquote
\renewenvironment{quote}[2][]
  {\if\relax\detokenize{#1}\relax
     \def\quoteauthor{#2}%
   \else
     \def\quoteauthor{#2~---~#1}%
   \fi
   \oldquote}
  {\par\nobreak\smallskip\hfill(\quoteauthor)%
   \endoldquote\addvspace{\bigskipamount}}


% set page margins and linespacing
\setlrmarginsandblock{3cm}{2cm}{*}
\setulmarginsandblock{2.5cm}{*}{1}
\linespread{1.1}
\checkandfixthelayout 


% set up header and footer for different page styles
\makepagestyle{rulednotitle}
\makeoddhead{rulednotitle}{}{}{\thepage}
\makeevenhead{rulednotitle}{\thepage}{}{}
\makeevenfoot{rulednotitle}{}{}{}
\makeoddfoot{rulednotitle}{}{}{}
\makeheadrule{rulednotitle}{\textwidth}{\normalrulethickness}

\makepagestyle{ruledtitle}
\makeevenhead{ruledtitle}{\thepage}{}{\scshape\MakeTextUppercase\leftmark}
\makeoddhead{ruledtitle}{\scshape\MakeTextUppercase\rightmark}{}{\thepage}
\makeoddfoot{ruledtitle}{}{}{}
\makeoddfoot{ruledtitle}{}{}{}
\makeheadrule{ruledtitle}{\textwidth}{\normalrulethickness}

% set up caption text
\captionsetup{labelfont=bf, textfont=sl, format=plain} % maybe small?
