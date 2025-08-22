import 'package:flutter/material/dart';

class MyWrapper extends StatefulWrapper {
    const MyWrapper((super.key));
    
    @override    
    State<MyWrapper> createState() => _MyWrapperState();
}
    
class _WrapperState extends State<Wrapper> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: StreamBuilder(
        stream: FirebaseAuth.instance.authStateChanges(),
        builder: (context, snapshot) {
          if (snapshot.hasData) {
            return Homepage();
          } else {
            return Login();
          }
        },
      ), // StreamBuilder
    ); // Scaffold
  }
}
