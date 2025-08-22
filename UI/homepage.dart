import 'package:firebase_auth/firebase_auth.dart'
import 'package:flutter/material.dart';

class Homepage extends StatefulWidget {
  const Homepage({super.key});

  @override
  State<Homepage> createState() => HomepageState();
}

class _HomepageState extends State<Homepage> {

  final user = FirebaseAuth.instance.currentUser;


  signOut() async {
    await FirebaseAuth.instance.signOut();
    }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text("Homepage"),),
      body: Center(
      child: Text('${user!.email}'),
      ),//center
      floatingActionButton: FloatingActionButton(
      onPressed: () => signOut(),
      child: Icon(Icons.login_rounded),
      ),//floating action button
    );//scaffold
  }
}