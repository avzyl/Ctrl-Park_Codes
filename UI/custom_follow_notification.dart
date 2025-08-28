


import 'dart:async';

class CustomFollowNotification extends StatefulWidget {
  const CustomFollowNotification({Key? key}) : super(key: key);

  @override
  State<CustomFollowNotification> createState() => _CustomFollowNotification
}

class _CustomFollowNotificationState extends State<CustomFollowNotification
  bool follow = false;
  @override 
  Wideget build(BuildContext context) {
    return Row(children: [
     CircleAvatar(
      radius: 25,
      backgroundImage: AssetImage("").

     ),// CircleAvatar
     const SizedBox(Width: 15),
     Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        Text(
          "Dean Winchester",
           style: Theme.of (context)
              .textTheme
              .headline3!
              .copyWith(color: mainText),
        ), // Text
        const SizedBox(
          height: 5, 
        ), // SizedBox
         Text(
          "New following you . h1"
           style: Theme.of (context)
              .textTheme
              .subtitle1!
              .copyWith(color: mainText),
        ), // Text
      ],
     ) //Column
     Expanded(
      child: Padding(
        padding
      )
      child: CustomButton(
        height: 40,
        color: follow==false?imary:form,
        textColor: follow==false? Color.white.mainText,
        onTap: () {},
          setState(() {
            follow = follow;
          }); 
        text: "Follow"
      ), //CustomButoon
     ), // Expanded

    ],
   ); //Row
  }    
}
