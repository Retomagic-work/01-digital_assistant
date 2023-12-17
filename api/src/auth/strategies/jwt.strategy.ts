import { ExtractJwt, Strategy } from "passport-jwt";
import { PassportStrategy } from "@nestjs/passport";
import { Injectable } from "@nestjs/common";
import { UserJwtDto } from "../../users/dto/user-jwt.dto";

@Injectable()
export class JwtStrategy extends PassportStrategy(Strategy) {
  constructor() {
    super({
      jwtFromRequest: ExtractJwt.fromAuthHeaderAsBearerToken(),
      ignoreExpiration: false,
      secretOrKey: process.env.JWT_SECRET_KEY || "secret",
    });
  }

  async validate(payload: any): Promise<UserJwtDto> {
    return { id: payload.sub, login: payload.username };
  }
}