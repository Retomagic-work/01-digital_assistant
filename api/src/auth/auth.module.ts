import { Module } from "@nestjs/common";
import { AuthService } from "./auth.service";
import { AuthController } from "./auth.controller";
import { UsersModule } from "../users/users.module";
import { JwtModule } from "@nestjs/jwt";
import { LocalStrategy } from "./strategies/local.auth";
import { JwtStrategy } from "./strategies/jwt.strategy";

@Module({
  imports: [
    JwtModule.register({
      secret: process.env.JWT_SECRET_KEY || "secret",
      signOptions: { expiresIn: "30d" }
    }),
    UsersModule
  ],
  providers: [AuthService, LocalStrategy, JwtStrategy],
  controllers: [AuthController],
  exports: [AuthService]
})
export class AuthModule {
}
