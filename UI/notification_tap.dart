import...

class NoticationTap extends StatelessWidget {
  const NoticationTap({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: Scaffold(
        body: Padding(padding: EdgeInsets.all(20),
        child: Column(
          children: [
            Text(
              "New",
              style: Theme.of(context).textTheme.headline1,
            ),//Text
            CustomFollowNotification(),
          ],
        ), //Column
       ), //Padding
     ), // Scaffold
    ); //SafeArea
  }
}