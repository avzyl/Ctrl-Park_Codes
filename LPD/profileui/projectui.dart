import...




class ProfileScreen extends StatelessWidget {
  const ProfileScreen({Key? key}) : super(key: key);

@override
Widget build(BuildContext context) {
  var isDark = MediaQuery.of(context).platformBrightness == Brightness.dark;
  return Scaffold(
    appBar: AppBar(
      leading: IconButton(
        onPressed: () {},
        icon: const Icon(LineAwesomeIcons.angle_left),
      ),
      title: Text(
        tProfile,
        style: Theme.of(context).textTheme.headline4,
      ),
      actions: [
        IconButton(
          onPressed: () {},
          icon: Icon(
            isDark ? LineAwesomeIcons.sun : LineAwesomeIcons.moon,
          ),
        ),
      ],
    ), //Appbar 
    body: SingleChildScrollView(
      child: Container(
        padding: const EdgeInsets.all(tDefaultSize),
        child: Column(
          children: [
            SizedBox(
              width: 120,
              height: 120,
              child: ClipRRect(
                borderRadius: BorderRadius.circular(100),child: const Image( image: AssetImage(tProfileImage))),
            ),//Sizedbox
            const SizedBox(height:10),
            Text(tProfileHeading, style: Theme.of(context).textTheme.headline4),
            Text(tProfileSubHeading, style:Theme.of(context).textTheme.bodytext2),
            const SizedBox(height: 20),
            SizedBox(
              width: 200,
              child ElevatedButton(
                onPressed: () {},
                child: const Text(tEditProfile, style: TextStyle(color: tDarkcolor)),
                style:ElevatedButton.stylefrom(
                  backgroundColor: tPrimaryColor, side:BorderSide.none, shape: const StadiumBorder()),
                child: const Text(tEditProfile,style: TextStyle(color: tDarkcolor)),
              ), //ElevatedButton
             ), //sizedbox
              const SizedBox(height: 30),
              const Divider(),
          ],
        ), //column
      ), //Container
    ), // SingleChildScrollView
  ); //Scaffold
}
